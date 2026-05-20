from pathlib import Path

from init_untrained_classifier import init_untrained_classifier
from model import RantFreeModel
from utils import get_info_from_env

def test_init_untrained_classifier():
    embedding_model_name_or_path, classifier_model_path = get_info_from_env()
    init_untrained_classifier(embedding_model_name_or_path, classifier_model_path)
    assert Path(classifier_model_path).exists()

def test_model_no_error():
    embedding_model_name_or_path, classifier_model_path = get_info_from_env()
    if not Path(classifier_model_path).exists():
        init_untrained_classifier(embedding_model_name_or_path, classifier_model_path)
    
    model = RantFreeModel(
        embedding_model_name_or_path=embedding_model_name_or_path,
        classifier_model_path=classifier_model_path,
    )

    texts = [
        "bro I’m honestly tired of always being the one putting effort into everything",
        "some people really talk like they know everything when they’re actually dumb as hell"
        "gila capek banget akhir akhir ini rasanya kepala penuh terus",
        "orang paling bacot biasanya paling ga ngerti apa apa",
        "最近まじで疲れた 何やってもうまくいかん感じする",
        "あいつ毎回偉そうだけど中身スカスカすぎて笑う",
        "akhir akhir iki mumet pol rasane pengen turu suwe wae",
        "omonge gedhe tenan ning otake kosong",
        "ayeuna teh cape pisan asa hayang rebahan wae sapoe",
        "loba gaya tapi euweuh eusi ngajadikeun geli sorangan",
    ]

    for text in texts:
        score_toxic, reason = model.predict(text)
        assert 0.0 <= score_toxic <= 1.0
        for r in reason:
            assert "token" in r
            assert isinstance(r["token"], str)
            assert "score" in r
            assert isinstance(r["score"], float)
