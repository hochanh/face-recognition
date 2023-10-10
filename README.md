# Simple Face Recognition

Using [insightface][1]

## Setup

```sh
pip install -r requirements.txt
python main.py
```

## Result

```txt
Ross 1 vs Ross 2: Same. Distance: 0.7504550814628601
Ross 1 vs Chandler 1: Different. Distance: 0.0229377131909132
Ross 1 vs Chandler 2: Different. Distance: 0.004771469160914421
Ross 2 vs Chandler 1: Different. Distance: 0.0328570231795311
Ross 2 vs Chandler 2: Different. Distance: 0.023528488352894783
Chandler 1 vs Chandler 2: Same. Distance: 0.4182615578174591
```



[1]: https://github.com/deepinsight/insightface