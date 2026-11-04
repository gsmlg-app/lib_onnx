#!/usr/bin/env python3
"""Audit real ELF PT_LOAD alignment and APK/AAB ZIP alignment (no SDK needed).

For APKs also run Android build-tools: zipalign -c -P 16 -v 4 app.apk.
AAB checks inspect ELF only; bundletool-generated APKs must be audited too.
"""
import argparse
import json
from pathlib import Path
import struct
import zipfile


def elf_check(data, name):
    if len(data) < 64 or data[:4] != b'\x7fELF' or data[4] != 2 or data[5] != 1:
        raise ValueError(f'{name}: expected little-endian ELF64')
    if struct.unpack_from('<H', data, 18)[0] != 183:
        raise ValueError(f'{name}: expected AArch64 ELF')
    phoff = struct.unpack_from('<Q', data, 32)[0]
    phentsize, phnum = struct.unpack_from('<HH', data, 54)
    if phentsize < 56 or not phnum or phoff + phentsize * phnum > len(data):
        raise ValueError(f'{name}: missing ELF program headers')
    segments = []
    for index in range(phnum):
        offset = phoff + index * phentsize
        kind, flags, fileoff, vaddr, paddr, filesz, memsz, align = struct.unpack_from('<IIQQQQQQ', data, offset)
        if kind == 1:
            if align < 16384 or (vaddr - fileoff) % 16384:
                raise ValueError(f'{name}: PT_LOAD[{index}] incompatible with 16 KB pages (align={align})')
            segments.append(align)
    if not segments:
        raise ValueError(f'{name}: no PT_LOAD segments')
    return {'name': name, 'load_alignments': segments, 'elf_16kb': 'passed'}


def audit(path):
    if path.suffix not in ('.apk', '.aab', '.zip'):
        return [elf_check(path.read_bytes(), str(path))]
    reports = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            with archive.open(info) as entry:
                is_elf = entry.read(4) == b'\x7fELF'
            if not is_elf and not info.filename.endswith('.so'):
                continue
            report = elf_check(archive.read(info), info.filename)
            if path.suffix == '.apk' and info.compress_type == zipfile.ZIP_STORED:
                with path.open('rb') as raw:
                    raw.seek(info.header_offset + 26)
                    filename_len, extra_len = struct.unpack('<HH', raw.read(4))
                data_offset = info.header_offset + 30 + filename_len + extra_len
                if data_offset % 16384:
                    raise ValueError(f'{info.filename}: uncompressed APK library ZIP offset {data_offset} not 16 KB aligned')
                report['apk_zip_16kb'] = 'passed'
            else:
                report['apk_zip_16kb'] = 'not-applicable-compressed' if path.suffix == '.apk' else 'requires-final-apk'
            reports.append(report)
    if not reports:
        raise ValueError(f'{path}: no packaged ELF files found')
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()
    reports = [report for path in args.files for report in audit(path)]
    print(json.dumps({'page_size_check': 'passed', 'binaries': reports}, indent=2))


if __name__ == '__main__':
    main()
