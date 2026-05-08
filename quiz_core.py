import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


QUIZ_SLOT_COUNT = 6
QUESTION_COUNT = 30
CHOICE_LABELS = ("A", "B", "C", "D")

DATA_DIR = Path("data")
QUIZ_DIR = DATA_DIR / "quizzes"
PROGRESS_DIR = DATA_DIR / "progress"
WRONG_ANSWERS_FILE = PROGRESS_DIR / "wrong_answers.json"
QUIZ_ID_RANGE = range(1, QUIZ_SLOT_COUNT + 1)


@dataclass
class Question:
    number: int
    prompt: str
    options: dict[str, str]
    correct_answers: list[str]
    explanation: str


@dataclass
class Quiz:
    quiz_id: int
    title: str
    questions: list[Question]


def ensure_data_dirs() -> None:
    QUIZ_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


def _compact_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def _normalize_block(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def _extract_question_sections(question_text: str) -> list[tuple[int, str]]:
    pattern = re.compile(
        r"(?ms)^\s*Question\s+(\d+)\s*:\s*(.*?)(?=^\s*Question\s+\d+\s*:|\Z)"
    )
    sections = []
    for match in pattern.finditer(question_text):
        number = int(match.group(1))
        body = match.group(2).strip()
        sections.append((number, body))
    return sections


def _extract_options(question_body: str) -> tuple[str, dict[str, str]]:
    option_pattern = re.compile(
        r"(?ms)^\s*([ABCD])(?:\.|\)|\s+-)\s*(.*?)(?=^\s*[ABCD](?:\.|\)|\s+-)\s|\Z)"
    )
    matches = list(option_pattern.finditer(question_body))
    if len(matches) != len(CHOICE_LABELS):
        raise ValueError("Khong tim du 4 lua chon A, B, C, D trong mot cau hoi.")

    prompt = question_body[: matches[0].start()].strip()
    options: dict[str, str] = {}
    for match in matches:
        label = match.group(1)
        text = _compact_whitespace(match.group(2).replace("\n", " "))
        options[label] = text

    missing = [label for label in CHOICE_LABELS if label not in options]
    if missing:
        raise ValueError(f"Thieu lua chon: {', '.join(missing)}")
    return _compact_whitespace(prompt.replace("\n", " ")), options


def _extract_answer_map(answer_text: str) -> dict[int, tuple[list[str], str]]:
    normalized = _compact_whitespace(answer_text.replace("\n", " "))
    pattern = re.compile(
        r"(\d+)\.\s*([A-D](?:\s*,\s*[A-D])*)\.\s*(.*?)(?=\s+\d+\.\s*[A-D](?:\s*,\s*[A-D])*\.\s*|$)"
    )
    answer_map: dict[int, tuple[list[str], str]] = {}
    for match in pattern.finditer(normalized):
        number = int(match.group(1))
        answers = [part.strip() for part in match.group(2).split(",")]
        explanation = match.group(3).strip()
        answer_map[number] = (answers, explanation)
    return answer_map


def parse_quiz_text(raw_text: str, quiz_id: int, title: str | None = None) -> Quiz:
    if quiz_id not in QUIZ_ID_RANGE:
        raise ValueError(f"So bai chi duoc tu 1 den {QUIZ_SLOT_COUNT}.")

    text = _normalize_block(raw_text)
    parts = re.split(r"(?is)ANSWER KEY\s*&\s*EXPLANATIONS", text, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("Khong tim thay phan ANSWER KEY & EXPLANATIONS.")

    question_text, answer_text = parts[0], parts[1]
    sections = _extract_question_sections(question_text)
    if len(sections) != QUESTION_COUNT:
        raise ValueError(f"Can {QUESTION_COUNT} cau hoi, parser dang tim thay {len(sections)} cau.")

    answer_map = _extract_answer_map(answer_text)
    if len(answer_map) != QUESTION_COUNT:
        raise ValueError(
            f"Can {QUESTION_COUNT} dap an, parser dang tim thay {len(answer_map)} cau trong answer key."
        )

    questions: list[Question] = []
    for expected_number, (number, body) in enumerate(sections, start=1):
        if number != expected_number:
            raise ValueError(f"Thu tu cau hoi khong dung: dang gap Question {number} thay vi {expected_number}.")

        prompt, options = _extract_options(body)
        if number not in answer_map:
            raise ValueError(f"Khong tim thay dap an cho cau {number}.")
        correct_answers, explanation = answer_map[number]
        questions.append(
            Question(
                number=number,
                prompt=prompt,
                options=options,
                correct_answers=correct_answers,
                explanation=explanation,
            )
        )

    quiz_title = title.strip() if title and title.strip() else f"Quiz {quiz_id}"
    return Quiz(quiz_id=quiz_id, title=quiz_title, questions=questions)


def save_quiz(quiz: Quiz) -> Path:
    ensure_data_dirs()
    path = QUIZ_DIR / f"quiz_{quiz.quiz_id}.json"
    payload = {
        "quiz_id": quiz.quiz_id,
        "title": quiz.title,
        "questions": [asdict(question) for question in quiz.questions],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_quiz(quiz_id: int) -> Quiz | None:
    path = QUIZ_DIR / f"quiz_{quiz_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Quiz(
        quiz_id=data["quiz_id"],
        title=data["title"],
        questions=[Question(**item) for item in data["questions"]],
    )


def quiz_slot_summary() -> list[dict[str, object]]:
    ensure_data_dirs()
    summary = []
    for quiz_id in QUIZ_ID_RANGE:
        quiz = load_quiz(quiz_id)
        summary.append(
            {
                "quiz_id": quiz_id,
                "title": quiz.title if quiz else f"Bai {quiz_id}",
                "ready": quiz is not None,
                "question_count": len(quiz.questions) if quiz else 0,
            }
        )
    return summary


def quiz_dropdown_choices() -> list[tuple[str, int]]:
    choices = []
    for item in quiz_slot_summary():
        suffix = "San sang" if item["ready"] else "Chua import"
        choices.append((f"Bai {item['quiz_id']} - {item['title']} ({suffix})", item["quiz_id"]))
    return choices


def load_wrong_answer_store() -> dict[str, list[int]]:
    ensure_data_dirs()
    if not WRONG_ANSWERS_FILE.exists():
        return {}
    return json.loads(WRONG_ANSWERS_FILE.read_text(encoding="utf-8"))


def save_wrong_answer_store(store: dict[str, list[int]]) -> None:
    ensure_data_dirs()
    WRONG_ANSWERS_FILE.write_text(
        json.dumps(store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def record_wrong_answers(quiz_id: int, wrong_numbers: list[int]) -> None:
    store = load_wrong_answer_store()
    key = str(quiz_id)
    merged = sorted(set(store.get(key, []) + wrong_numbers))
    store[key] = merged
    save_wrong_answer_store(store)


def clear_wrong_answers(quiz_id: int) -> None:
    store = load_wrong_answer_store()
    store[str(quiz_id)] = []
    save_wrong_answer_store(store)


def load_wrong_questions(quiz_id: int) -> list[int]:
    store = load_wrong_answer_store()
    return store.get(str(quiz_id), [])


def build_progress_markdown() -> str:
    lines = [
        "# Tong quan 6 bai quiz",
        "",
    ]
    for item in quiz_slot_summary():
        wrong_count = len(load_wrong_questions(int(item["quiz_id"])))
        lines.append(
            f"- Bai {item['quiz_id']}: {item['title']} | "
            f"{'San sang' if item['ready'] else 'Chua import'} | "
            f"{item['question_count']}/{QUESTION_COUNT} cau | "
            f"{wrong_count} cau dang can on"
        )
    return "\n".join(lines)
