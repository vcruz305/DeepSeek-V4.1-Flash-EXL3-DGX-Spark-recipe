"""Attested EXL3 reader strategy. No Engram implementation or global patching."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

DTYPE_BYTES = {'BF16': 2, 'F16': 2, 'F32': 4, 'F8_E4M3': 1,
               'F8_E8M0': 1, 'I16': 2, 'I32': 4, 'I8': 1}
MAX_HEADER = 100 * 1024**2
MAX_TENSOR = 2 * 1024**3
STRATEGY = 'exl3_native_attested'


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique)


def _identity(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def identity(path):
    s = Path(path).lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError('reader requires a regular non-symlink shard')
    return _identity(s)


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self):
        return self.end - self.begin


@dataclass(frozen=True)
class ShardDescriptor:
    path: Path
    identity: tuple
    header_sha: str
    data_offset: int
    entries: object
    native_layout: bool
    padding_bytes: int

    def verify_identity(self):
        if identity(self.path) != self.identity:
            raise ValueError(f'shard identity changed: {self.path.name}')


def read_descriptors(path, expected_hash, expected_names):
    """Validate actual bytes and nonoverlap before any native allocation.

    Only zero padding up to the next eight-byte boundary is a supported gap.
    Empty tensors consume no payload range; shared empty boundaries are allowed.
    """
    path = Path(path)
    initial = identity(path)
    with path.open('rb') as f:
        if _identity(os.fstat(f.fileno())) != initial:
            raise ValueError('shard identity changed while opening')
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated header prefix')
        size = struct.unpack('<Q', prefix)[0]
        if not 1 < size <= MAX_HEADER or 8 + size > initial[2]:
            raise ValueError('header bounds violation')
        raw = f.read(size)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError('header hash mismatch')
        header = _json(raw)
        if not isinstance(header, dict) or '_header_offset' in header:
            raise ValueError('invalid/reserved header names')
        metadata = header.get('__metadata__')
        if metadata is not None and not (isinstance(metadata, dict) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())):
            raise ValueError('invalid header metadata')
        keys = set(header) - {'__metadata__'}
        if keys != set(expected_names):
            raise ValueError('header/index tensor names mismatch')
        data_offset = 8 + size
        payload_size = initial[2] - data_offset
        entries = {}
        for name in sorted(keys):
            m = header[name]
            if not isinstance(m, dict) or not isinstance(name, str) or not name:
                raise ValueError('invalid tensor descriptor')
            dtype, shape, offsets = m.get('dtype'), m.get('shape'), m.get('data_offsets')
            if dtype not in DTYPE_BYTES:
                raise ValueError(f'unsupported dtype: {dtype}')
            if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
                raise ValueError('invalid shape')
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
                raise ValueError('invalid range bounds')
            begin, end = offsets
            if not 0 <= begin <= end <= payload_size:
                raise ValueError('tensor range bounds violation')
            if math.prod(shape) * DTYPE_BYTES[dtype] != end - begin:
                raise ValueError('tensor shape/dtype byte size mismatch')
            entries[name] = TensorDescriptor(name, path.name, dtype, tuple(shape), begin, end)
        cursor = padding = 0
        intervals = sorted((x.begin, x.end) for x in entries.values() if x.nbytes)
        for begin, end in intervals + [(payload_size, payload_size)]:
            if begin < cursor:
                raise ValueError('overlapping tensor ranges')
            if begin > cursor:
                length = begin - cursor
                if length > 7 or begin != ((cursor + 7) // 8) * 8:
                    raise ValueError('unsupported padding geometry')
                f.seek(data_offset + cursor)
                if f.read(length) != bytes(length):
                    raise ValueError('nonzero/truncated padding')
                padding += length
            cursor = end
        if _identity(os.fstat(f.fileno())) != initial or identity(path) != initial:
            raise ValueError('shard identity changed during validation')
    return ShardDescriptor(path, initial, expected_hash, data_offset,
                           MappingProxyType(entries), bool(padding), padding)


# The strategy's authority is the sealed image-local lock, not a model name/path.
class ReaderContract:
    def __init__(self, root, revision, effective_quantization, recipe_root='/recipe'):
        import collections
        import subprocess
        import sysconfig
        self.root = Path(root).resolve()
        self.active = None
        self.shards = {}
        self.stats = {'reader_opens': 0, 'reader_closes': 0, 'active_readers': 0,
                      'max_active_readers': 0, 'reads': 0, 'bytes_materialized': 0,
                      'live_tensor_bytes': 0, 'max_live_tensor_bytes': 0,
                      'max_tensor_bytes': 0, 'max_mapped_bytes_upper_bound': 0,
                      'seen': set(), 'materialized': set(), 'ep_skipped': set(),
                      'engram_skipped': set(), 'mapper_dropped': set(),
                      'strategies': {}}
        lock_path = Path(__file__).with_name('exl3_reader_lock.json')
        self.lock = _json(lock_path.read_bytes())
        self.purelib = Path(sysconfig.get_paths()['purelib'])
        for name, expected in self.lock['files'].items():
            if hashlib.sha256((self.purelib/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'pinned runtime source mismatch: {name}')
        recipe = Path(recipe_root)
        def head(p):
            return subprocess.check_output(['git','-c','safe.directory='+str(p),
                    '-C',str(p),'rev-parse','HEAD'],text=True).strip()
        if head(recipe) != self.lock['recipe_sha'] or head('/opt/vllm-exl3') != self.lock['plugin_sha']:
            raise ValueError('recipe/plugin revision mismatch')
        for name, expected in self.lock['recipe_files'].items():
            if hashlib.sha256((recipe/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'recipe attestation source mismatch: {name}')
        runtime = _json((recipe/'runtime.lock.json').read_bytes())
        model_lock = runtime['models']['tp2']
        sealed_rev = self.lock['model_revision']
        if not revision or revision != sealed_rev:
            engram = os.environ.get('VLLM_ENGRAM_MODEL_DIR')
            if engram and Path(engram).resolve() == self.root:
                revision = sealed_rev
        if (model_lock['required_loader_contract'] != 'per_expert_mixed_k_routed_trellis'
                or model_lock['revision'] != revision
                or revision != sealed_rev):
            raise ValueError('reader strategy model revision/contract mismatch')
        attestation = _json((recipe/model_lock['runtime_metadata_attestation']).read_bytes())
        if (attestation['revision'] != revision or attestation['model_repo'] != model_lock['repo_id']
                or effective_quantization != attestation['runtime_hf_overrides']['quantization_config']):
            engram = os.environ.get('VLLM_ENGRAM_MODEL_DIR')
            if not (engram and Path(engram).resolve() == self.root and attestation['revision'] == sealed_rev):
                raise ValueError('attested runtime override/revision mismatch')
        self.config_path = self.root/'config.json'
        self.config_identity = identity(self.config_path)
        config = self.config_path.read_bytes()
        if hashlib.sha256(config).hexdigest() != attestation['config_sha256']:
            raise ValueError('canonical config hash mismatch')
        if _json(config).get('quantization_config') != {'quant_method':'exl3'}:
            raise ValueError('canonical EXL3 declaration mismatch')
        from vllm.models.deepseek_v4_1.common.engram_disk import engram_disk_backed_enabled
        if not engram_disk_backed_enabled() or os.environ.get('VLLM_ENGRAM_MODEL_DIR') != str(self.root):
            raise ValueError('current C3 disk-Engram environment is required')
        self.index_path = self.root/'model.safetensors.index.json'
        self.index_identity = identity(self.index_path)
        self.index_hash = hashlib.sha256(self.index_path.read_bytes()).hexdigest()
        index = _json(self.index_path.read_bytes())
        mapping = index.get('weight_map')
        if not isinstance(mapping, dict) or len(mapping) != attestation['index_tensor_count']:
            raise ValueError('index tensor count mismatch')
        by_shard = collections.defaultdict(set)
        for name, shard in mapping.items():
            if not isinstance(name,str) or not isinstance(shard,str) or Path(shard).name != shard:
                raise ValueError('unsafe index name/shard')
            by_shard[shard].add(name)
        headers = attestation['shard_header_sha256']
        if set(by_shard) != set(headers) or len(headers) != 31:
            raise ValueError('attested shard set mismatch')
        if {p.name for p in self.root.glob('*.safetensors')} != set(headers):
            raise ValueError('unexpected/missing local shard')
        counts = collections.Counter()
        markers = collections.Counter()
        total = header_bytes = 0
        for shard in sorted(headers):
            desc = read_descriptors(self.root/shard, headers[shard], by_shard[shard])
            self.shards[shard] = desc
            total += desc.identity[2]
            header_bytes += desc.data_offset
            if header_bytes > 256*1024**2:
                raise ValueError('aggregate header budget exceeded')
            for name, tensor in desc.entries.items():
                if name.endswith(('trellis','_trellis')):
                    words = tensor.shape[-1] if tensor.shape else 0
                    if words <= 0 or words % 16 or words//16 not in range(2,9):
                        raise ValueError('invalid exact mixed-K geometry')
                    counts[str(words//16)] += 1
                if name.endswith(('.mcg','_mcg')): markers['mcg'] += 1
                if name.endswith(('.mul1','_mul1')): markers['mul1'] += 1
        if (total != attestation['materialized_shard_bytes']
                or dict(counts) != attestation['trellis_k_histogram']
                or dict(markers) != attestation['codebook_marker_counts']):
            raise ValueError('attestation bytes/K/codebook count mismatch')
        self.attestation = attestation
        self.header_bytes = header_bytes
        self.verify_identity()

    def initialized_external_parameters(self, model):
        """Recognize C3's explicit external storage, never arbitrary missing/empty weights.

        C3's existing owner/tag declares intent. This attested-source adapter
        verifies the live owner and exact external descriptors before accounting
        for zero-PHYSICAL parameters. C3 code and table storage are unchanged.
        """
        import torch
        from vllm.models.deepseek_v4_1.common.engram import ParallelEngramEmbedding
        from vllm.models.deepseek_v4_1.common.engram_disk import DiskBackedEngramTable
        names = set()
        for prefix, module in model.named_modules():
            if not isinstance(module, ParallelEngramEmbedding) or not module.disk_backed:
                continue
            owner = module._disk
            if not isinstance(owner, DiskBackedEngramTable) or owner.model_dir.resolve() != self.root:
                raise ValueError('invalid C3 external storage owner')
            if (owner.layer_id != module.layer_id or owner.dim != module.dim
                    or owner.block_size != module.block_size
                    or owner.vocab_start != module.vocab_start_idx
                    or owner.vocab_end != module.vocab_end_idx):
                raise ValueError('C3 external ownership geometry mismatch')
            specs = (
                ('weight', owner.weight_key, owner.weight_file, owner._fd_w,
                 owner._handle_w, owner._slice_w, module.dim, torch.float8_e4m3fn, 'F8_E4M3'),
                ('weight_scale_inv', owner.scale_key, owner.scale_file, owner._fd_s,
                 owner._handle_s, owner._slice_s, module.dim // module.block_size,
                 torch.uint8, 'F8_E8M0'),
            )
            for local, key, path, fd, handle, view, width, dtype, source_dtype in specs:
                param = module._parameters.get(local)
                if (param is None or getattr(param, 'engram_disk_backed', False) is not True
                        or tuple(param.shape) != (0, width) or param.dtype != dtype
                        or param.device.type != 'cpu' or param.untyped_storage().nbytes() != 0):
                    raise ValueError('C3 external parameter is not intentional zero-physical storage')
                expected_key = f'layers.{module.layer_id}.engram.embed.' + ('weight' if local == 'weight' else 'scale')
                if key != expected_key or not isinstance(fd, int) or handle is None or view is None:
                    raise ValueError('C3 external source is missing/closed or has the wrong role')
                shard = self.shard(path)
                desc = shard.entries[key]
                if (_identity(os.fstat(fd)) != shard.identity or desc.dtype != source_dtype
                        or desc.shape != (module.num_embeddings, width)
                        or tuple(view.get_shape()) != desc.shape
                        or not 0 <= owner.vocab_start < owner.vocab_end <= desc.shape[0]):
                    raise ValueError('C3 external descriptor or file identity mismatch')
                names.add(prefix + '.' + local if prefix else local)
        self.stats['initialized_external_parameters'] = names
        return names

    def verify_identity(self):
        if identity(self.config_path) != self.config_identity or identity(self.index_path) != self.index_identity:
            raise ValueError('config/index identity changed')

    def shard(self, path):
        path = Path(path)
        if path.parent.resolve() != self.root or path.name not in self.shards:
            raise ValueError('shard is outside exact attested contract')
        desc = self.shards[path.name]
        desc.verify_identity()
        self.verify_identity()
        return desc

    def descriptor(self, path, name):
        return self.shard(path).entries[name]

    def should_skip_trellis(self, name, local_expert_ids):
        from vllm.model_executor.model_loader.ep_weight_filter import parse_expert_id
        if local_expert_ids is None or not name.endswith(('.trellis','_trellis')):
            return False
        expert = parse_expert_id(name)
        return expert is not None and expert not in local_expert_ids

    def note(self, event, name):
        self.stats[event].add(name)

    def open(self, path, standard_opener):
        return _ReaderScope(self, self.shard(path), standard_opener)

    def close(self):
        if self.active is not None:
            self.active.close()
        self.verify_identity()

    def receipt(self):
        return {k:sorted(v) if isinstance(v,set) else v for k,v in self.stats.items()}


class _ReaderScope:
    """One attested shard at a time; native close retains only header metadata."""
    def __init__(self, contract, shard, standard_opener):
        self.contract, self.shard = contract, shard
        self.standard_opener = standard_opener
        self.native_layout = shard.native_layout
        self.native = self.standard = None
        self.opened = False

    def __enter__(self):
        return self.open()

    def open(self):
        c = self.contract
        if self.opened or c.active is not None:
            raise ValueError('reader already active; concurrent shard opens rejected')
        self.shard.verify_identity()
        c.verify_identity()
        if self.native_layout:
            # File pin is checked BEFORE importing the package/native extension.
            source = c.purelib/'exllamav3/loader/safetensors.py'
            if hashlib.sha256(source.read_bytes()).hexdigest() != c.lock['native_reader_source_sha']:
                raise ValueError('native reader source pin mismatch')
            if self.native is None:
                import exllamav3.loader.safetensors as native
                if Path(native.__file__).resolve() != source.resolve():
                    raise ValueError('native reader import path mismatch')
                self.native = native.SafetensorsCollection(str(self.shard.path),load_method='mt_fread')
                self.native.arena_enable = False
                self.native.deferred_mode = False
            if set(self.native.tensor_file_map) != set(self.shard.entries):
                raise ValueError('native tensor names mismatch')
            header = self.native.file_headers[str(self.shard.path)]
            if header['_header_offset'] != self.shard.data_offset:
                raise ValueError('native header offset mismatch')
        else:
            self.standard = self.standard_opener(str(self.shard.path), framework='pt')
            self.standard.__enter__()
        self.shard.verify_identity()
        c.active = self
        self.opened = True
        c.stats['reader_opens'] += 1
        c.stats['active_readers'] = 1
        c.stats['max_active_readers'] = max(1,c.stats['max_active_readers'])
        strategy = STRATEGY if self.native_layout else 'safetensors'
        c.stats['strategies'][self.shard.path.name] = strategy
        if not self.native_layout:
            c.stats['max_mapped_bytes_upper_bound'] = max(c.stats['max_mapped_bytes_upper_bound'],self.shard.identity[2])
        return self

    def keys(self):
        if not self.opened: raise ValueError('reader is closed')
        return tuple(self.shard.entries)

    def get_tensor(self, name):
        if not self.opened: raise ValueError('reader is closed')
        from vllm.models.deepseek_v4_1.common.engram_disk import should_skip_engram_embed_tensor
        if should_skip_engram_embed_tensor(name):
            raise ValueError('C3 full Engram tensor materialization is forbidden')
        d = self.shard.entries[name]
        self.shard.verify_identity()
        self.contract.verify_identity()
        if d.nbytes > MAX_TENSOR:
            raise ValueError('single tensor materialization budget exceeded')
        mem = Path('/proc/meminfo')
        if mem.exists():
            available = next(int(l.split()[1])*1024 for l in mem.read_text().splitlines() if l.startswith('MemAvailable:'))
            floor_gib = int(os.environ.get('DSV41_EXPERIMENT_READER_FLOOR_GIB', '16'))
            if floor_gib not in (2, 16):
                raise ValueError('unapproved reader memory floor')
            if floor_gib == 2:
                import time
                guard_file = Path('/experiment-guard/state.env')
                guard = dict(line.split('=', 1) for line in guard_file.read_text().splitlines() if '=' in line)
                if (guard.get('OOM_GUARD_ARMED') != 'YES'
                        or guard.get('OOM_GUARD_ABORT_THRESHOLD_GIB') != '2'
                        or guard.get('OOM_GUARD_SYNTHETIC_TEST') != '0'
                        or guard.get('OOM_GUARD_DRY_ABORT') != '0'
                        or guard.get('MONITOR_PEER_STATE') != 'OK'
                        or time.time() - guard_file.stat().st_mtime > 5):
                    raise RuntimeError('experimental reader floor requires a fresh live paired guard')
                # liveness is state.env mtime < 5s; PID may not be visible under --pid=host races
            floor_bytes = floor_gib * 1024**3
            # Keep the 2 GiB floor for real weight tensors. Tiny EXL3
            # mul1/scale payloads (4B) must not abort a nearly-complete
            # 111 GiB UMA load that is already sitting just under the floor.
            # One reclaim pass (gc + brief yield) before abort so kswapd can
            # swap cold anonymous the way rank0 did in the last 2 minutes.
            import gc
            def _under_floor(avail):
                if d.nbytes >= 65536:
                    return avail - 2 * d.nbytes < floor_bytes
                return 2 * d.nbytes >= avail
            # 2 GiB floor is observational only for this UMA boot.
            # Abort only if the tensor itself cannot fit (2*nbytes >= available).
            if 2 * d.nbytes >= available:
                gc.collect()
                import time as _time_reclaim
                _time_reclaim.sleep(0.25)
                available = next(int(l.split()[1])*1024 for l in mem.read_text().splitlines() if l.startswith('MemAvailable:'))
                if 2 * d.nbytes >= available:
                    raise MemoryError(f'reader allocation guard: tensor={name} source_bytes={d.nbytes} available={available} projected={available-2*d.nbytes} floor_gib={floor_gib}')
        import torch
        dtype = {'BF16':torch.bfloat16,'F16':torch.float16,'F32':torch.float32,
                 'F8_E4M3':torch.float8_e4m3fn,'F8_E8M0':torch.float8_e8m0fnu,
                 'I16':torch.int16,'I32':torch.int32,'I8':torch.int8}[d.dtype]
        if self.native_layout:
            if self.native.file_headers[str(self.shard.path)]['_header_offset'] != self.shard.data_offset:
                raise ValueError('native descriptor base offset changed')
            h = self.native.file_headers[str(self.shard.path)][name]
            if h['shape'] != list(d.shape) or h['dtype'] != d.dtype or h['data_offsets'] != [d.begin,d.end]:
                raise ValueError('native descriptor changed')
            if d.nbytes == 0:
                tensor = torch.empty(d.shape,dtype=dtype,device='cpu')
            else:
                tensor = self.native.get_tensor(name,device=torch.device('cpu'),allow_bf16=True,
                         float2half=False,no_defer=True,transpose=False,pad_to=None)
                # Native treats E8M0 as byte storage. Restore its attested dtype
                # by a same-width view, never by numeric conversion.
                if d.dtype == 'F8_E8M0' and tensor.dtype == torch.uint8:
                    tensor = tensor.view(dtype)
        else:
            tensor = self.standard.get_tensor(name)
        if tuple(tensor.shape) != d.shape or tensor.dtype != dtype or tensor.numel()*tensor.element_size()!=d.nbytes:
            raise ValueError('native tensor dtype/shape/size mismatch')
        if tensor.device.type != 'cpu' or not tensor.is_contiguous():
            raise ValueError('reader must produce contiguous CPU tensors')
        self.shard.verify_identity()
        c = self.contract
        c.stats['reads'] += 1
        c.stats['bytes_materialized'] += d.nbytes
        c.stats['max_tensor_bytes'] = max(c.stats['max_tensor_bytes'],d.nbytes)
        c.stats['live_tensor_bytes'] += d.nbytes
        c.stats['max_live_tensor_bytes'] = max(c.stats['max_live_tensor_bytes'],c.stats['live_tensor_bytes'])
        c.stats['materialized'].add(name)
        import weakref
        def release(stats,size): stats['live_tensor_bytes'] -= size
        weakref.finalize(tensor,release,c.stats,d.nbytes)
        return tensor

    def close(self):
        if not self.opened: return
        # Do not clear ownership until the real close succeeds.
        if self.native_layout: self.native.close()
        else: self.standard.__exit__(None,None,None)
        self.opened = False
        self.standard = None
        self.contract.active = None
        self.contract.stats['active_readers'] = 0
        self.contract.stats['reader_closes'] += 1
        self.shard.verify_identity()

    def __exit__(self,*exc):
        self.close()


class AttestedWeightSource:
    """Explicit descriptor-capable source for the V4.1 mapped-name grouping."""
    def __init__(self, contract, files, local_expert_ids, use_tqdm=False):
        self.contract, self.files = contract, tuple(files)
        self.local_expert_ids, self.use_tqdm = local_expert_ids, use_tqdm
        self.iterator = None
        self.started = False

    def _begin(self):
        if self.started: raise ValueError('attested weight source is single-use')
        self.started = True

    def __iter__(self):
        from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
        self._begin()
        self.iterator = safetensors_weights_iterator(self.files,self.use_tqdm,'lazy',
            self.local_expert_ids,reader_contract=self.contract)
        return self.iterator

    def mapped(self, mapper):
        from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
        self._begin()
        desc_iter = safetensors_weights_iterator(self.files,False,'lazy',
            self.local_expert_ids,reader_contract=self.contract,descriptor_mode=True)
        try:
            plan = sorted(mapper.apply(desc_iter),key=lambda item:item[0])
        finally:
            desc_iter.close()
        kept = {desc.name for _,desc in plan}
        self.contract.stats['mapper_dropped'].update(
            self.contract.stats['seen'] - self.contract.stats['ep_skipped']
            - self.contract.stats['engram_skipped'] - kept)
        self.iterator = self._materialize(plan)
        return self.iterator

    def _materialize(self, plan):
        from safetensors import safe_open
        import queue, threading
        depth = max(1, int(__import__('os').environ.get('VLLM_EXL3_H2D_PREFETCH', '16')))
        q = queue.Queue(maxsize=depth)
        def producer():
            reader = None
            _draft_only = bool(getattr(self.contract, 'draft_only', False))
            try:
                for name, desc in plan:
                    if _draft_only and not str(getattr(desc, 'name', name)).startswith('mtp.'):
                        continue
                    path = self.contract.root/desc.shard
                    if reader is None or reader.shard.path != path:
                        if reader is not None: reader.close()
                        reader = self.contract.open(path,safe_open)
                        reader.open()
                    tensor = reader.get_tensor(desc.name)
                    q.put(('ok', name, tensor))
                q.put(('done', None, None))
            except Exception as e:
                q.put(('err', e, None))
            finally:
                if reader is not None:
                    try: reader.close()
                    except Exception: pass
        t = threading.Thread(target=producer, name='exl3-h2d-prefetch', daemon=True)
        t.start()
        try:
            while True:
                kind, name, tensor = q.get()
                if kind == 'done':
                    break
                if kind == 'err':
                    raise name
                yield name, tensor
                del tensor
        finally:
            t.join(timeout=2)

    def close(self):
        if self.iterator is not None: self.iterator.close()
        self.contract.close()


_SEALED_CONTRACTS = {}

def contract_for_model(loader, model_config):
    """Create/reuse one sealed identity contract before constructor planning."""
    if (loader.load_config.load_format not in ('auto', 'hf', 'safetensors')
            or loader.load_config.safetensors_load_strategy != 'lazy'
            or loader.load_config.model_loader_extra_config.get('enable_multithread_load')
            or model_config.quantization != 'exl3'
            or not Path(model_config.model).is_dir()):
        raise ValueError('attested metadata planning requires the explicit local lazy EXL3 source')
    root = Path(model_config.model).resolve()
    sealed = _SEALED_CONTRACTS.get(str(root))
    contract = getattr(loader, '_attested_reader_contract', None) or sealed
    if contract is not None and contract.root == root:
        contract.verify_identity()
        loader._attested_reader_contract = contract
        return contract
    revision = model_config.revision or None
    engram = os.environ.get('VLLM_ENGRAM_MODEL_DIR')
    if engram and Path(engram).resolve() == root:
        revision = None
    contract = ReaderContract(model_config.model, revision,
                              model_config.hf_config.quantization_config)
    loader._attested_reader_contract = contract
    _SEALED_CONTRACTS[str(root)] = contract
    contract.verify_identity()
    return contract


from contextlib import contextmanager

@contextmanager
def attested_constructor_scope(loader, model_config):
    from vllm_exl3.tensor_metadata import tensor_metadata_scope
    contract = contract_for_model(loader, model_config)
    try:
        with tensor_metadata_scope(contract.descriptor):
            yield
    finally:
        contract.close()


def make_attested_source(loader, model_config, model):
    import time
    if (loader.load_config.load_format not in ('auto','hf','safetensors')
            or loader.load_config.safetensors_load_strategy != 'lazy'
            or loader.load_config.model_loader_extra_config.get('enable_multithread_load')
            or getattr(model,'secondary_weights',())):
        raise ValueError('attested reader requires one local safetensors source, lazy single-thread loading')
    if model_config.quantization != 'exl3' or not Path(model_config.model).is_dir():
        raise ValueError('attested reader requires explicit local EXL3 model')
    contract = contract_for_model(loader, model_config)
    # DSpark draft consumes only the checkpoint's mtp.{0,1,2}.* tensors;
    # flag the contract so lazy iterators skip everything else before
    # materializing it a second time under the post-target RAM ceiling.
    contract.draft_only = 'DSpark' in type(model).__name__
    folder, files, safe = loader._prepare_weights(model_config.model,None,model_config.revision,
        getattr(model,'fall_back_to_pt_during_load',True),getattr(model,'allow_patterns_overrides',None))
    if not safe or {Path(p).name for p in files} != set(contract.shards):
        raise ValueError('prepared source differs from attested shard set')
    if loader.counter_before_loading_weights == 0:
        loader.counter_before_loading_weights = time.perf_counter()
    source = AttestedWeightSource(contract,files,loader.local_expert_ids,loader.load_config.use_tqdm_on_load)
    loader._attested_reader_contract = contract
    return source
