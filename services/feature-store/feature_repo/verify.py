from feast import FeatureStore

store = FeatureStore(repo_path=r'C:\Users\Batrisyia Zahrani\Documents\01 ITB\rantfree\feature_repo\feature_repo')
features = store.get_online_features(
    features=['comment_features:text_length', 'comment_features:toxic_word_count'],
    entity_rows=[{'id': '0000997932d777bf'}]
).to_dict()
print(features)
