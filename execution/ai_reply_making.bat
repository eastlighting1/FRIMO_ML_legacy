@ECHO OFF

cd C:\Users\User\Desktop\model\KoGPT2-chatbot
python train_torch.py --gpus 1 --chat --user_input %1