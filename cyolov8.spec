# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['cyolov8.py'],
    pathex=[],
    binaries=[
        ('C:\\Users\\PJ\\python_venv\\CYOLOv8_venv\\Lib\\site-packages\\cv2\\opencv_videoio_ffmpeg490_64.dll', '.')  # OpenCVのDLLファイルを追加
    ],
    datas=[
        ('C:\\Users\\PJ\\python_venv\\CYOLOv8_venv\\Lib\\site-packages\\ultralytics\\cfg\\default.yaml', 'ultralytics\\cfg'),
        ('model/*', 'model'), ('model_descriptions.csv', '.')
    ],
    hiddenimports=['cv2', 'PIL.Image', 'ultralytics'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='cyolov8',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='cyolov8'
)
