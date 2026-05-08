import gradio as gr

from quiz_core import (
    Question,
    Quiz,
    ensure_data_dirs,
    list_available_quizzes,
    load_quiz,
    load_wrong_questions,
    parse_quiz_text,
    record_wrong_answers,
    save_quiz,
)


QUESTION_SLOT_COUNT = 30


def preview_quiz_import(raw_text: str, quiz_id: int, title: str) -> tuple[str, str]:
    quiz = parse_quiz_text(raw_text, quiz_id, title)
    preview_lines = [f"# {quiz.title}", f"Da nhan {len(quiz.questions)} cau hoi.", ""]
    for question in quiz.questions[:3]:
        preview_lines.append(f"Question {question.number}: {question.prompt}")
        for label in "ABCD":
            preview_lines.append(f"- {label}. {question.options[label]}")
        preview_lines.append(f"Dap an: {', '.join(question.correct_answers)}")
        preview_lines.append("")
    preview = "\n".join(preview_lines).strip()
    return (
        f"Parse thanh cong. San sang luu bai {quiz.quiz_id} voi {len(quiz.questions)} cau.",
        preview,
    )


def save_quiz_import(raw_text: str, quiz_id: int, title: str) -> tuple[str, str]:
    quiz = parse_quiz_text(raw_text, quiz_id, title)
    path = save_quiz(quiz)
    options = _quiz_dropdown_choices()
    return f"Da luu {quiz.title} vao {path}", gr.update(choices=options, value=quiz.quiz_id)


def _quiz_dropdown_choices() -> list[tuple[str, int]]:
    choices = list_available_quizzes()
    if choices:
        return choices
    return [("Chua co bai nao duoc import", 0)]


def _question_block(question: Question, checked: list[str], show_result: bool) -> str:
    selected = set(checked)
    correct = set(question.correct_answers)
    status = ""
    if show_result:
        status = " Dung." if selected == correct else " Sai."

    lines = [f"### Cau {question.number}{status}", question.prompt, ""]
    for label in "ABCD":
        marker = "[x]" if label in selected else "[ ]"
        answer_hint = ""
        if show_result and label in correct:
            answer_hint = " <- dap an dung"
        lines.append(f"{marker} {label}. {question.options[label]}{answer_hint}")
    if show_result:
        lines.append("")
        lines.append(f"Giai thich: {question.explanation}")
    return "\n".join(lines)


def _selected_questions(quiz_id: int, review_wrong_only: bool) -> tuple[Quiz, list[Question]]:
    quiz = load_quiz(quiz_id)
    if quiz is None:
        raise gr.Error("Chua tim thay bai quiz da luu.")

    questions = quiz.questions
    if review_wrong_only:
        wrong_numbers = set(load_wrong_questions(quiz_id))
        questions = [question for question in quiz.questions if question.number in wrong_numbers]
    return quiz, questions


def _render_quiz(quiz_id: int, review_wrong_only: bool):
    quiz, questions = _selected_questions(quiz_id, review_wrong_only)
    if review_wrong_only and not questions:
        updates = [f"# {quiz.title}", "Khong co cau sai da luu cho bai nay."]
        for _ in range(QUESTION_SLOT_COUNT):
            updates.extend(
                [
                    gr.update(value="", visible=False),
                    gr.update(choices=["A", "B", "C", "D"], value=[], visible=False),
                ]
            )
        return updates

    note = (
        f"Dang tai che do on {len(questions)} cau sai."
        if review_wrong_only
        else f"Dang tai day du {len(questions)} cau."
    )
    updates = [f"# {quiz.title}", note]
    for idx in range(QUESTION_SLOT_COUNT):
        if idx < len(questions):
            question = questions[idx]
            body = f"### Cau {question.number}\n{question.prompt}\n\nA. {question.options['A']}\n\nB. {question.options['B']}\n\nC. {question.options['C']}\n\nD. {question.options['D']}"
            updates.extend(
                [
                    gr.update(value=body, visible=True),
                    gr.update(choices=["A", "B", "C", "D"], value=[], visible=True),
                ]
            )
        else:
            updates.extend(
                [
                    gr.update(value="", visible=False),
                    gr.update(choices=["A", "B", "C", "D"], value=[], visible=False),
                ]
            )
    return updates


def submit_answers(quiz_id: int, review_wrong_only: bool, *answers) -> tuple[str, str]:
    quiz, selected_questions = _selected_questions(quiz_id, review_wrong_only)
    if review_wrong_only and not selected_questions:
        raise gr.Error("Khong co cau sai nao de cham trong che do on lai.")

    result_lines = [f"# Ket qua {quiz.title}", ""]
    wrong_numbers: list[int] = []
    correct_count = 0

    for idx, question in enumerate(selected_questions):
        chosen = sorted(set(answers[idx] or []))
        if set(chosen) == set(question.correct_answers):
            correct_count += 1
        else:
            wrong_numbers.append(question.number)
        result_lines.append(_question_block(question, chosen, show_result=True))
        result_lines.append("")

    total = len(selected_questions)
    score_line = f"Diem: {correct_count}/{total} ({(correct_count / total) * 100:.1f}%)"
    summary = score_line
    if wrong_numbers:
        record_wrong_answers(quiz_id, wrong_numbers)
        summary += f"\nCau sai: {', '.join(str(num) for num in wrong_numbers)}"
    else:
        summary += "\nBan da dung toan bo trong pham vi bai dang lam."

    return summary, "\n".join(result_lines).strip()


ensure_data_dirs()
initial_choices = _quiz_dropdown_choices()
initial_quiz = initial_choices[0][1] if initial_choices else 0
render_outputs = []

with gr.Blocks(title="Multiple Select Quiz App") as demo:
    gr.Markdown(
        """
        # Multiple Select Quiz App

        Import de thi tu raw text, lam bai theo kieu multiple select, va on lai cac cau da sai.
        """
    )

    with gr.Tab("Import Quiz"):
        import_quiz_id = gr.Dropdown(
            choices=[(f"Bai {idx}", idx) for idx in range(1, 7)],
            value=1,
            label="So bai quiz",
        )
        import_title = gr.Textbox(label="Tieu de", placeholder="Vi du: Luat bat dong san Phap - Bai 1")
        raw_input = gr.Textbox(label="Paste raw text", lines=24)
        with gr.Row():
            preview_button = gr.Button("Parse Preview")
            save_button = gr.Button("Luu Quiz", variant="primary")
        import_status = gr.Markdown()
        preview_output = gr.Markdown()

    with gr.Tab("Lam Bai"):
        quiz_selector = gr.Dropdown(
            choices=initial_choices,
            value=initial_quiz,
            label="Chon bai quiz",
        )
        review_wrong_only = gr.Checkbox(label="Chi on cau sai da luu", value=False)
        with gr.Row():
            load_button = gr.Button("Tai Bai")
            submit_button = gr.Button("Cham Bai", variant="primary")
        quiz_view = gr.Markdown()
        quiz_status = gr.Markdown()
        answer_components: list[gr.CheckboxGroup] = []
        question_components: list[gr.Markdown] = []
        for slot in range(QUESTION_SLOT_COUNT):
            question_components.append(gr.Markdown(visible=False))
            answer_components.append(
                gr.CheckboxGroup(
                    choices=["A", "B", "C", "D"],
                    label=f"Chon dap an cau {slot + 1}",
                    visible=False,
                )
            )
            render_outputs.extend([question_components[-1], answer_components[-1]])
        score_output = gr.Markdown()
        result_view = gr.Markdown()

    preview_button.click(
        fn=preview_quiz_import,
        inputs=[raw_input, import_quiz_id, import_title],
        outputs=[import_status, preview_output],
    )
    save_button.click(
        fn=save_quiz_import,
        inputs=[raw_input, import_quiz_id, import_title],
        outputs=[import_status, quiz_selector],
    )
    load_button.click(
        fn=_render_quiz,
        inputs=[quiz_selector, review_wrong_only],
        outputs=[quiz_view, quiz_status, *render_outputs],
    )
    submit_button.click(
        fn=submit_answers,
        inputs=[quiz_selector, review_wrong_only, *answer_components],
        outputs=[score_output, result_view],
    )


if __name__ == "__main__":
    demo.launch()
