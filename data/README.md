# 数据集说明

### seq.jsonl
数据样例：
```json
[
    [480, 53961, null, {"112": 19, "117": 103, "118": 125, "119": 87, "120": 126, "100": 2, "101": 40, "102": 8559, "122": 5998, "114": 16, "116": 1, "121": 52176, "111": 5630}, 0, 1746077791], 
    [480, 50654, null, {"112": 15, "117": 285, "118": 737, "119": 1500, "120": 1071, "100": 6, "101": 22, "102": 10420, "122": 2269, "114": 16, "115": 43, "116": 13, "121": 12734, "111": 6737}, 0, 1746094091], 
    [480, 23149, null, {"112": 15, "117": 84, "118": 774, "119": 1668, "120": 348, "100": 6, "101": 32, "102": 6372, "122": 2980, "114": 16, "116": 15, "121": 30438, "111": 34195}, 0, 1746225104],
    ...
]
```
1. 每一行为一个用户的行为序列，按时间排序
2. 用户序列中每一个record的数据格式为：
```
[user_id, item_id, user_feature, item_feature, action_type, timestamp]
```
*注：
    i. 每一条record为记录user profile和item interaction其中的一个。若当前record为user profile，则item_id、item_feature、action_type为null；若当前record为item interaction，则user_feature为null。
    ii. user_feature/item_feature的值以dict组织，其中key表示feature_id，value表示对应的feature_value取值。
3. user_id, item_id, feature_value全部从1进行了重新编号（re-id），方便在模型中进行embedding lookup。原始值到re-id的映射参考下方的indexer.pkl文件。

### indexer.pkl
记录了原始wuid, creative_id和feature_value从1开始编号后的值（以下记为re-id）
```
indexer['u'] # 记录了wuid到re-id,key为wuid,value为re-id
indexer['i'] # 记录了creative_id到re-id,key为creative_id,value为re-id
indexer['f'] # 记录了各feature_id中feature_value到re-id，例如：⬇️
indexer['f']['112'] # 记录了feature_id 112中的feature_value到re-id的映射
```

### item_feat_dict.json
1. 记录了训练集中所有item的feature，方便训练时负采样。
2. key为item re-id，value为item的feature

### seq_offsets.pkl
记录了seq.jsonl中每一行的起始文件指针offset，方便读取数据时快速定位数据位置，提升I/O效率。
