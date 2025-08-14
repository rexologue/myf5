import subprocess
import os

# Define paths
PATH_TO_CKPT  = "/home/user/f5/ckpts_full/model_417600.pt" #"/home/user/audio/f5/ckpts/model_last.pt" #"/home/user/audio/f5/ckpts1/model_426150.pt"
PATH_TO_VOCAB = "/home/user/f5/project/vocab.txt"
REF_PATH      = "/home/user/f5/project/refs"
CFG           = "F5TTS_v1_Base"
DEVICE        = "cuda" #"cpu"

# Infer stuff
OUTPUT_PATH = "/home/user/f5/out"
# Text to generate
TEXT_TO_GENERATE = "Нет, для ст+арта не нужн+а. Бенеф+ит от т+очных транскр+иптов в+ыше, чем от разм+етки п+ауз. Если п+озже зах+очется — м+ожно доб+авить прост+ые пр+авки."
# TEXT_TO_GENERATE = "Ал+иса так+ая каз+явка - пр+осто нет слов. Никогд+а еще не видел+а наст+олько коз+явочную де+вочку!"
TEXT_TO_GENERATE = (
    "Мал+инки,  мал+инки,  так+ие вечер+инки  "
    "Зел+ёные троп+инки,  где т+ихо и свеж+о  "
    "Мал+инки,  мал+инки,  брюн+етки и блонд+инки  "
    "Сер+ёжки и Мар+инки  "
    "Эй, дидж+ей, заряж+ай пласт+инки!"
)

TEXT_TO_GENERATE = "Здр+авствуйте! Мен+я зов+ут Екатер+ина, я представ+итель компани+и Орифл+эйм. Подскаж+ите,   вам уд+обно сейч+ас разгов+аривать? "

# TEXT_TO_GENERATE = "По ит+огу разгов+ора мы прих+одим к том+у, что Крым в пр+инципе им не подх+одит, потом+у что там у с+ервиса, ну на д+анный мом+ент, так+ого, к кот+орому он+и прив+ыкли пр+осто нет."

# Read reference text
with open(os.path.join(REF_PATH, "ref.txt"), "r", encoding="utf-8") as f:
    REF_TEXT = f.read().strip()

# Change directory to F5_TTS/src
os.chdir("F5_TTS/src")

# Build the command
command = [
    "python",
    "-m",
    "f5_tts.infer.infer_cli",
    "-p", PATH_TO_CKPT,
    "-m", CFG,
    "-v", PATH_TO_VOCAB,
    "-r", os.path.join(REF_PATH, "ref.wav"),
    "-s", REF_TEXT,
    "-t", TEXT_TO_GENERATE,
    "-o", OUTPUT_PATH,
    "--device", DEVICE
]

# Run the command
try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running the command: {e}")
except Exception as e:
    print(f"An error occurred: {e}")