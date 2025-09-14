import subprocess
import os

from formater import format_text

# Define paths
PATH_TO_CKPT  = "/root/ckpt/model_last.pt"
PATH_TO_VOCAB = "/root/ckpt/vocab.txt"
REF_PATH      = "/root/myf5/f5/project/refs1"
CFG           = "F5TTS_v1_Base"
DEVICE        = "cuda" #"cpu"

# Infer stuff
OUTPUT_PATH = "/root/myf5/f5/out"
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

# TEXT_TO_GENERATE = "Здр+авствуйте! Мен+я зов+ут Екатер+ина, я представ+итель компани+и Орифл+эйм. Подскаж+ите,   вам уд+обно сейч+ас разгов+аривать? "

TEXT_TO_GENERATE = "По ит+огу разгов+ора мы прих+одим к том+у, что Крым в пр+инципе им не подх+одит, потом+у что там у с+ервиса, ну на д+анный мом+ент, так+ого, к кот+орому он+и прив+ыкли пр+осто нет."

TEXT_TO_GENERATE = "Вам было бы интересно, если бы я предложила подобный вариант?"

# Read reference text
with open(os.path.join(REF_PATH, "ref.txt"), "r", encoding="utf-8") as f:
    REF_TEXT = f.read().strip()

# Change directory to F5_TTS/src
os.chdir("F5_TTS/src")

# Build the command
command = [
    "/root/.venv/bin/python",
    "-m",
    "f5_tts.infer.infer_cli",
    "-p", PATH_TO_CKPT,
    "-m", CFG,
    "-v", PATH_TO_VOCAB,
    "-r", os.path.join(REF_PATH, "ref.wav"),
    "-s", REF_TEXT,
    "-t", format_text(TEXT_TO_GENERATE),
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