from faster_whisper import WhisperModel

_model = None

def get_model():
    global _model
    if _model is None:
        _model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8",
            cpu_threads=8,
        )
    return _model
