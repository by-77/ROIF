

## Train
If you need to train  from scratch, you should use commond line as following.

      python main.py new -n name -d data-dir -b batch-size -e epochs  --n noise

     python main.py new --name ftcarw -d C:\Users\... -b 4 -e 60  --noise  'cropout((0.55, 0.6), (0.55, 0.6))'

Environmental requirements:
+ Python == 3.7.4; Torch == 1.12.1 + cu102; Torchvision == 0.13.1; PIL == 7.2.0

## Test
Put the pre-trained model into pretrain floder, and you can test ROIF by command line as following.

      python test.py -o ./pretrain/options-and-config.pickle -c ./pretrain/ROIF.pyt -s data-dir -n noise
 


