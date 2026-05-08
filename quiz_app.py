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

QUIZIZZ_CSS = """
:root {
  --quiz-bg: radial-gradient(circle at top, #3b236f 0%, #1b1236 42%, #110a24 100%);
  --quiz-card: rgba(255, 255, 255, 0.08);
  --quiz-border: rgba(255, 255, 255, 0.16);
  --quiz-text: #f8f7ff;
  --quiz-muted: #d7cffc;
  --quiz-accent: #ffcf33;
  --quiz-pink: #ff5fa2;
  --quiz-cyan: #40d9ff;
  --quiz-green: #5be08b;
  --quiz-orange: #ff9c40;
}

.gradio-container {
  background: var(--quiz-bg);
  color: var(--quiz-text);
}

.gradio-container .block-container {
  max-width: 1200px;
  padding-top: 24px;
}

.quizizz-shell {
  background: linear-gradient(135deg, rgba(255, 95, 162, 0.16), rgba(64, 217, 255, 0.12));
  border: 1px solid var(--quiz-border);
  border-radius: 28px;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
  padding: 28px;
}

.quizizz-hero {
  background: linear-gradient(135deg, rgba(255, 207, 51, 0.16), rgba(255, 95, 162, 0.18));
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 24px;
  margin-bottom: 20px;
  padding: 24px;
}

.quizizz-hero h1 {
  color: white;
  font-size: 2.1rem;
  margin: 0 0 10px;
}

.quizizz-hero p {
  color: var(--quiz-muted);
  font-size: 1rem;
  margin: 0;
}

.quizizz-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 16px;
}

.quizizz-badge {
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 999px;
  color: white;
  font-size: 0.95rem;
  font-weight: 700;
  padding: 8px 14px;
}

.quizizz-shell .tab-nav {
  background: rgba(255, 255, 255, 0.06);
  border-radius: 18px;
  padding: 8px;
}

.quizizz-shell .tab-nav button {
  border-radius: 14px !important;
  color: white !important;
  font-weight: 700 !important;
}

.quizizz-shell .tab-nav button.selected {
  background: linear-gradient(135deg, #ff5fa2, #7f6bff) !important;
}

.quizizz-shell .gr-box,
.quizizz-shell .gr-group,
.quizizz-shell .gr-form,
.quizizz-shell .gr-panel,
.quizizz-shell .gr-accordion {
  background: var(--quiz-card) !important;
  border: 1px solid var(--quiz-border) !important;
  border-radius: 22px !important;
}

.quizizz-shell textarea,
.quizizz-shell input,
.quizizz-shell .wrap,
.quizizz-shell .scroll-hide {
  color: white !important;
}

.quizizz-shell label span,
.quizizz-shell .prose,
.quizizz-shell .prose p,
.quizizz-shell .prose li {
  color: var(--quiz-text) !important;
}

.quizizz-shell .primary {
  background: linear-gradient(135deg, #ff5fa2, #ff8f40) !important;
  border: 0 !important;
}

.quizizz-shell button {
  border-radius: 16px !important;
  font-weight: 800 !important;
}

.quizizz-status {
  background: rgba(91, 224, 139, 0.16);
  border: 1px solid rgba(91, 224, 139, 0.42);
  border-radius: 16px;
  color: white;
  padding: 14px 16px;
}

.quizizz-question-card {
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.11), rgba(255, 255, 255, 0.04));
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 22px;
  box-shadow: 0 14px 40px rgba(0, 0, 0, 0.22);
  margin-top: 12px;
  padding: 18px 20px;
}

.quizizz-question-card .q-chip {
  background: linear-gradient(135deg, #ffcf33, #ff8f40);
  border-radius: 999px;
  color: #28114c;
  display: inline-block;
  font-size: 0.88rem;
  font-weight: 900;
  padding: 6px 12px;
}

.quizizz-question-card h3 {
  color: white;
  font-size: 1.18rem;
  line-height: 1.45;
  margin: 14px 0 0;
}

.quizizz-options .wrap {
  display: grid !important;
  gap: 12px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.quizizz-options label {
  align-items: center !important;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  border-radius: 18px !important;
  min-height: 74px;
  padding: 16px 14px !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}

.quizizz-options label:hover {
  border-color: rgba(255, 255, 255, 0.34) !important;
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.2);
  transform: translateY(-2px);
}

.quizizz-options label:nth-child(1) { background: rgba(255, 95, 162, 0.18); }
.quizizz-options label:nth-child(2) { background: rgba(64, 217, 255, 0.18); }
.quizizz-options label:nth-child(3) { background: rgba(91, 224, 139, 0.18); }
.quizizz-options label:nth-child(4) { background: rgba(255, 156, 64, 0.18); }

.quizizz-options input:checked + span,
.quizizz-options input:checked ~ span {
  color: white !important;
  font-weight: 800;
}

.quizizz-options label:has(input:checked) {
  border-color: rgba(255, 255, 255, 0.56) !important;
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.12), 0 16px 32px rgba(0, 0, 0, 0.24);
}

@media (max-width: 800px) {
  .quizizz-options .wrap {
    grid-template-columns: 1fr;
  }
}
"""


def _question_card_html(question: Question) -> str:
    return (
        '<div class="quizizz-question-card">'
        f'<span class="q-chip">Cau {question.number}</span>'
        f"<h3>{question.prompt}</h3>"
        "</div>"
    )


def _status_html(message: str) -> str:
    return f'<div class="quizizz-status">{message}</div>'


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
    updates = [title, _status_html(note)]
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
    updates = [f"# {quiz.title}", _status_html(note)]
    for idx in range(QUESTION_COUNT):
        if idx < len(questions):
            question = questions[idx]
            updates.extend(
                [
                    gr.update(value=_question_card_html(question), visible=True),
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

with gr.Blocks(title="Multiple Select Quiz App", css=QUIZIZZ_CSS) as demo:
    with gr.Column(elem_classes=["quizizz-shell"]):
        gr.HTML(
            """
            <section class="quizizz-hero">
              <h1>Multiple Select Quiz Arena</h1>
              <p>Phong cach quiz game de hoc va on tap 6 bai de, moi bai 30 cau, co cham nhanh va review cau sai.</p>
              <div class="quizizz-badges">
                <span class="quizizz-badge">6 Bai Kiem Tra</span>
                <span class="quizizz-badge">30 Cau Moi Bai</span>
                <span class="quizizz-badge">Multiple Select</span>
                <span class="quizizz-badge">Review Loi Sai</span>
              </div>
            </section>
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
            quiz_view = gr.Markdown(elem_classes=["quizizz-title"])
            quiz_status = gr.HTML()
            answer_components: list[gr.CheckboxGroup] = []
            question_components: list[gr.HTML] = []
            for slot in range(QUESTION_COUNT):
                question_components.append(gr.HTML(visible=False))
                answer_components.append(
                    gr.CheckboxGroup(
                        choices=list(CHOICE_LABELS),
                        label=f"Chon dap an cau {slot + 1}",
                        visible=False,
                        elem_classes=["quizizz-options"],
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
