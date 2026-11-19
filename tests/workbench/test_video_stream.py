import shutil
import subprocess

import pytest

from omega_playground.ops.video import FragmentedVideo


def box(kind, body):
    return (len(body) + 8).to_bytes(4, 'big') + kind + body


def test_partial_boxes_and_bounded_complete_fragments(tmp_path):
    stream = FragmentedVideo(tmp_path / 'video.mp4')
    init = box(b'ftyp', b'format') + box(b'moov', b'header')
    fragments = [box(b'moof', bytes([i])) + box(b'mdat', bytes([i])*17) for i in range(15)]
    data = init + b''.join(fragments)
    for i in range(0, len(data), 3):
        stream.feed(data[i:i+3])
    assert stream.init == init
    assert list(stream.fragments) == list(enumerate(fragments[5:], 6))
    assert stream.sequence == 15


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg required')
def test_late_join_fragments_decode_without_prior_frames(tmp_path):
    path = tmp_path / 'video.mp4'
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'lavfi', '-i',
                    'testsrc=size=64x64:rate=30:duration=2', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-tune', 'zerolatency', '-profile:v', 'baseline',
                    '-g', '15', '-sc_threshold', '0', '-movflags',
                    'frag_keyframe+empty_moov+default_base_moof', str(path)], check=True)
    stream = FragmentedVideo(path)
    stream.feed(path.read_bytes())
    assert len(stream.fragments) == 4
    joined = tmp_path / 'late.mp4'
    joined.write_bytes(stream.init + b''.join(data for _, data in list(stream.fragments)[-2:]))
    result = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(joined),
                             '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                            check=True, capture_output=True)
    assert len(result.stdout) == 30 * 64 * 64 * 3
