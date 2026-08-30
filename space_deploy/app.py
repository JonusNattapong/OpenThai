"""
Interactive Web Application for OpenThai-NER (Gradio).
Ready for Hugging Face Spaces deployment.
"""

import gradio as gr
from openthai_ner import OpenThaiNER
from openthai_ner.utils import render_html_highlight

# Initialize model pipeline lazily
ner_pipeline = None


def get_pipeline():
    global ner_pipeline
    if ner_pipeline is None:
        ner_pipeline = OpenThaiNER()
    return ner_pipeline


def analyze_text(text: str, confidence_threshold: float):
    if not text or not text.strip():
        return "<p>กรุณากรอกข้อความเพื่อวิเคราะห์</p>", []

    pipeline = get_pipeline()
    entities = pipeline.predict(text, threshold=confidence_threshold)

    # HTML highlighted visualization
    html_output = render_html_highlight(text, entities)

    # Table records
    table_records = [
        [ent["word"], ent["entity"], ent["start"], ent["end"], f"{ent['score']:.2%}"]
        for ent in entities
    ]

    return html_output, table_records


EXAMPLES = [
    ["นายสมชาย เข็มกลัด เดินทางไปประชุมที่กระทรวงการคลัง ถนนพระราม 6 ในวันที่ 15 มกราคม 2568 เวลา 10:30 น. เพื่อลงนามสัญญาจัดซื้อระบบคอมพิวเตอร์มูลค่า 15,000,000 บาท กับบริษัท ไทยซอฟต์แวร์ จำกัด", 0.5],
    ["กรมควบคุมโรค เตือนประชาชนระวังการระบาดของโรคไข้เลือดออกในพื้นที่กรุงเทพมหานครและปริมณฑล สอบถามเพิ่มเติมสายด่วน 1422 หรืออีเมล contact@ddc.mail.go.th", 0.5],
    ["ธนาคารแห่งประเทศไทย ประกาศปรับลดอัตราดอกเบี้ยนโยบายลง 0.25% เพื่อกระตุ้นเศรษฐกิจและการลงทุนในประเทศ", 0.5],
]

demo = gr.Blocks(
    title="OpenThai-NER Demo",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
)

with demo:
    gr.Markdown(
        """
        # 🇹🇭 OpenThai-NER: Thai Named Entity Recognition
        
        โมเดลประมวลผลและสกัดข้อมูลเฉพาะ (Named Entities) ในภาษาไทย รองรับหลากหลายหมวดหมู่:
        `PERSON`, `ORGANIZATION`, `LOCATION`, `DATE`, `TIME`, `MONEY`, `PERCENT`, `LAW`, `PHONE`, `EMAIL`, `URL` ฯลฯ
        
        *โมเดลพัฒนาโดย [JonusNattapong/OpenThai-NER](https://huggingface.co/JonusNattapong/OpenThai-NER)*
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            input_text = gr.Textbox(
                label="ป้อนข้อความภาษาไทย (Thai Input Text)",
                lines=5,
                placeholder="พิมพ์ข้อความที่ต้องการวิเคราะห์ที่นี่...",
            )
            threshold_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Confidence Threshold (เกณฑ์ความเชื่อมั่น)",
            )
            submit_btn = gr.Button("วิเคราะห์ข้อความ (Extract Entities)", variant="primary")

        with gr.Column(scale=4):
            gr.Markdown("### ผลลัพธ์การสกัด Entity (Highlighted Output)")
            html_display = gr.HTML(label="Highlighted Text")
            gr.Markdown("### รายละเอียดข้อมูลเอนทิตี (Entity Table)")
            table_display = gr.Dataframe(
                headers=["คำ / วลี (Entity Text)", "ประเภท (Label)", "Start Index", "End Index", "ความเชื่อมั่น (Confidence)"],
                datatype=["str", "str", "number", "number", "str"],
                label="Entities Extracted",
            )

    submit_btn.click(
        fn=analyze_text,
        inputs=[input_text, threshold_slider],
        outputs=[html_display, table_display],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[input_text, threshold_slider],
        outputs=[html_display, table_display],
        fn=analyze_text,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()
