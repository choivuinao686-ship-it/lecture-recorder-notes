import gradio as gr

from quiz_core import (
    CHOICE_LABELS,
    QUESTION_COUNT,
    QUIZ_ID_RANGE,
    Question,
    Quiz,
    build_progress_markdown,
    clear_wrong_answers,
    ensure_data_dirs,
    load_quiz,
    load_wrong_questions,
    parse_quiz_text,
    quiz_dropdown_choices,
    record_wrong_answers,
    save_quiz,
)


def preview_quiz_import(raw_text: str, quiz_id: int, title: str) -> tuple[str, str, str]:
    quiz = parse_quiz_text(raw_text, quiz_id, title)
    preview_lines = [f"# {quiz.title}", f"Da nhan {len(quiz.questions)} cau hoi.", ""]
    for question in quiz.questions[:3]:
        preview_lines.append(f"Question {question.number}: {question.prompt}")
        for label in CHOICE_LABELS:
            preview_lines.append(f"- {label}. {question.options[label]}")
        preview_lines.append(f"Dap an: {', '.join(question.correct_answers)}")
        preview_lines.append("")
    preview = "\n".join(preview_lines).strip()
    return (
        f"Parse thanh cong. Bai {quiz.quiz_id} da san sang de luu.",
        preview,
        build_progress_markdown(),
    )


def save_quiz_import(raw_text: str, quiz_id: int, title: str):
    quiz = parse_quiz_text(raw_text, quiz_id, title)
    path = save_quiz(quiz)
    choices = quiz_dropdown_choices()
    status = f"Da luu {quiz.title} vao {path}."
    return (
        status,
        gr.update(choices=choices, value=quiz.quiz_id),
        gr.update(choices=choices, value=quiz.quiz_id),
        build_progress_markdown(),
    )


def _question_block(question: Question, checked: list[str], show_result: bool) -> str:
    selected = set(checked)
    correct = set(question.correct_answers)
    status = ""
    if show_result:
        status = " Dung." if selected == correct else " Sai."

    lines = [f"### Cau {question.number}{status}", question.prompt, ""]
    for label in CHOICE_LABELS:
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
        raise gr.Error("Bai nay chua duoc import. Hay paste raw text o tab Import Quiz truoc.")

    questions = quiz.questions
    if review_wrong_only:
        wrong_numbers = set(load_wrong_questions(quiz_id))
        questions = [question for question in quiz.questions if question.number in wrong_numbers]
    return quiz, questions


def _empty_quiz_render(title: str, note: str):
    updates = [title, note]
    for _ in range(QUESTION_COUNT):
        updates.extend(
            [
                gr.update(value="", visible=False),
                gr.update(choices=list(CHOICE_LABELS), value=[], visible=False),
            ]
        )
    return updates


def render_quiz(quiz_id: int, review_wrong_only: bool):
    quiz, questions = _selected_questions(quiz_id, review_wrong_only)
    if review_wrong_only and not questions:
        return _empty_quiz_render(
            f"# {quiz.title}",
            "Khong co cau sai da luu cho bai nay. Ban co the lam bai thuong truoc.",
        )

    note = (
        f"Dang tai che do on {len(questions)} cau sai."
        if review_wrong_only
        else f"Dang tai day du {len(questions)} cau."
    )
    updates = [f"# {quiz.title}", note]
    for idx in range(QUESTION_COUNT):
        if idx < len(questions):
            question = questions[idx]
            body = (
                f"### Cau {question.number}\n{question.prompt}\n\n"
                f"A. {question.options['A']}\n\n"
                f"B. {question.options['B']}\n\n"
                f"C. {question.options['C']}\n\n"
                f"D. {question.options['D']}"
            )
            updates.extend(
                [
                    gr.update(value=body, visible=True),
                    gr.update(choices=list(CHOICE_LABELS), value=[], visible=True),
                ]
            )
        else:
            updates.extend(
                [
                    gr.update(value="", visible=False),
                    gr.update(choices=list(CHOICE_LABELS), value=[], visible=False),
                ]
            )
    return updates


def _grade_answers(
    quiz_id: int,
    review_wrong_only: bool,
    partial_only: bool,
    *answers,
) -> tuple[str, str, str]:
    quiz, selected_questions = _selected_questions(quiz_id, review_wrong_only)
    if review_wrong_only and not selected_questions:
        raise gr.Error("Khong co cau sai nao de cham trong che do on lai.")

    result_lines = [f"# Ket qua {quiz.title}", ""]
    wrong_numbers: list[int] = []
    correct_count = 0
    attempted_count = 0
    skipped_numbers: list[int] = []

    for idx, question in enumerate(selected_questions):
        chosen = sorted(set(answers[idx] or []))
        if partial_only and not chosen:
            skipped_numbers.append(question.number)
            continue

        attempted_count += 1
        if set(chosen) == set(question.correct_answers):
            correct_count += 1
        else:
            wrong_numbers.append(question.number)
        result_lines.append(_question_block(question, chosen, show_result=True))
        result_lines.append("")

    if partial_only and attempted_count == 0:
        raise gr.Error("Ban chua chon dap an nao, nen app chua co gi de cham.")

    total = len(selected_questions)
    denominator = attempted_count if partial_only else total
    score_line = f"Diem: {correct_count}/{denominator} ({(correct_count / denominator) * 100:.1f}%)"
    summary = score_line
    if partial_only:
        summary += f"\nDa cham {attempted_count}/{total} cau da tra loi."
        if skipped_numbers:
            summary += f"\nBo qua: {', '.join(str(num) for num in skipped_numbers)}"
    if wrong_numbers:
        record_wrong_answers(quiz_id, wrong_numbers)
        summary += f"\nCau sai: {', '.join(str(num) for num in wrong_numbers)}"
    else:
        if review_wrong_only:
            clear_wrong_answers(quiz_id)
        summary += "\nBan da dung toan bo trong pham vi bai dang lam."

    return summary, "\n".join(result_lines).strip(), build_progress_markdown()


def reset_review_list(quiz_id: int) -> tuple[str, str]:
    quiz = load_quiz(quiz_id)
    if quiz is None:
        raise gr.Error("Bai nay chua duoc import nen khong co danh sach sai de xoa.")
    clear_wrong_answers(quiz_id)
    return f"Da xoa danh sach cau sai cua {quiz.title}.", build_progress_markdown()


def submit_answers(quiz_id: int, review_wrong_only: bool, *answers) -> tuple[str, str, str]:
    return _grade_answers(quiz_id, review_wrong_only, False, *answers)


def stop_and_grade_answers(quiz_id: int, review_wrong_only: bool, *answers) -> tuple[str, str, str]:
    return _grade_answers(quiz_id, review_wrong_only, True, *answers)


ensure_data_dirs()
initial_choices = quiz_dropdown_choices()
initial_quiz = 1
render_outputs = []

with gr.Blocks(title="Multiple Select Quiz App") as demo:
    gr.Markdown(
        """
        # Multiple Select Quiz App

        App nay dung cho 6 bai kiem tra, moi bai 30 cau, multiple select, va co che do on lai cau sai.
        """
    )

    with gr.Tab("Import Quiz"):
        gr.Markdown("Moi bai quiz la 1 raw input gom 30 cau va 1 answer key. Ban import lan luot cho Bai 1 den Bai 6.")
        import_quiz_id = gr.Dropdown(
            choices=[(f"Bai {idx}", idx) for idx in QUIZ_ID_RANGE],
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
        review_wrong_only = gr.Checkbox(label="Chi on cac cau da sai cua bai nay", value=False)
        with gr.Row():
            load_button = gr.Button("Tai Bai")
            submit_button = gr.Button("Cham Bai", variant="primary")
            reset_wrong_button = gr.Button("Xoa danh sach cau sai")
            stop_button = gr.Button("Dung va cham phan da lam")
        quiz_view = gr.Markdown()
        quiz_status = gr.Markdown()
        answer_components: list[gr.CheckboxGroup] = []
        question_components: list[gr.Markdown] = []
        for slot in range(QUESTION_COUNT):
            question_components.append(gr.Markdown(visible=False))
            answer_components.append(
                gr.CheckboxGroup(
                    choices=list(CHOICE_LABELS),
                    label=f"Chon dap an cau {slot + 1}",
                    visible=False,
                )
            )
            render_outputs.extend([question_components[-1], answer_components[-1]])
        score_output = gr.Markdown()
        result_view = gr.Markdown()

    with gr.Tab("Tien Do"):
        progress_quiz_id = gr.Dropdown(
            choices=initial_choices,
            value=initial_quiz,
            label="Bai can thao tac progress",
        )
        progress_output = gr.Markdown(value=build_progress_markdown())

    preview_button.click(
        fn=preview_quiz_import,
        inputs=[raw_input, import_quiz_id, import_title],
        outputs=[import_status, preview_output, progress_output],
    )
    save_button.click(
        fn=save_quiz_import,
        inputs=[raw_input, import_quiz_id, import_title],
        outputs=[import_status, quiz_selector, progress_quiz_id, progress_output],
    )
    load_button.click(
        fn=render_quiz,
        inputs=[quiz_selector, review_wrong_only],
        outputs=[quiz_view, quiz_status, *render_outputs],
    )
    submit_button.click(
        fn=submit_answers,
        inputs=[quiz_selector, review_wrong_only, *answer_components],
        outputs=[score_output, result_view, progress_output],
    )
    reset_wrong_button.click(
        fn=reset_review_list,
        inputs=[quiz_selector],
        outputs=[score_output, progress_output],
    )
    stop_button.click(
        fn=stop_and_grade_answers,
        inputs=[quiz_selector, review_wrong_only, *answer_components],
        outputs=[score_output, result_view, progress_output],
    )


if __name__ == "__main__":
    demo.launch()
