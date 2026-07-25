import os
import sys
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable, KeepTogether

REPORT_MD = r"p:\project\hackothon\jnn_shivamogga\report.md"
REPORT_PDF = r"p:\project\hackothon\jnn_shivamogga\report.pdf"

def build_pdf():
    doc = SimpleDocTemplate(
        REPORT_PDF,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1a252f'),
        spaceAfter=12,
        alignment=0
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceBefore=14,
        spaceAfter=8
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#34495e'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'Callout_Custom',
        parent=body_style,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1b4f72'),
        backColor=colors.HexColor('#eaf2f8'),
        borderColor=colors.HexColor('#2980b9'),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=6,
        spaceAfter=6
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#273746'),
        backColor=colors.HexColor('#f8f9f9'),
        borderColor=colors.HexColor('#bdc3c7'),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=6
    )

    th_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    tb_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#2c3e50'),
        alignment=1
    )

    story = []

    with open(REPORT_MD, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_table = False
    table_lines = []

    in_code = False
    code_buffer = []

    def clean_text(text):
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # Bold
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # Italic
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        # Inline code
        text = re.sub(r'`(.*?)`', r'<font face="Courier" color="#c0392b">\1</font>', text)
        # LaTeX math simple fix
        text = text.replace('$', '')
        return text

    for line in lines:
        raw_line = line
        line = line.strip()

        # Handle Code Block
        if line.startswith('```'):
            if in_code:
                in_code = False
                code_text = '\n'.join(code_buffer)
                story.append(Paragraph(clean_text(code_text).replace('\n', '<br/>'), code_style))
                code_buffer = []
            else:
                in_code = True
                code_buffer = []
            continue

        if in_code:
            code_buffer.append(raw_line.rstrip('\n'))
            continue

        # Handle Markdown Tables
        if '|' in line:
            in_table = True
            table_lines.append(line)
            continue
        elif in_table:
            in_table = False
            # Render accumulated table
            rows = []
            for tline in table_lines:
                if '---' in tline:
                    continue
                cells = [c.strip() for c in tline.split('|')[1:-1]]
                rows.append(cells)
            
            if rows:
                table_data = []
                # Header
                header_row = [Paragraph(clean_text(c), th_style) for c in rows[0]]
                table_data.append(header_row)
                # Body
                for r in rows[1:]:
                    row_cells = [Paragraph(clean_text(c), tb_style) for c in r]
                    table_data.append(row_cells)

                t = Table(table_data, hAlign='LEFT')
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('LEFTPADDING', (0, 0), (-1, -1), 4),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f4f4')])
                ]))
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 6))
            table_lines = []

        if not line:
            story.append(Spacer(1, 4))
            continue

        # Images
        img_match = re.match(r'!\[.*?\]\((file:///)?(.*?)\)', line)
        if img_match:
            img_path = img_match.group(2)
            if os.path.exists(img_path):
                img = Image(img_path, width=480, height=270)
                story.append(Spacer(1, 6))
                story.append(img)
                story.append(Spacer(1, 6))
            continue

        # Horizontal Rule
        if line == '---':
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#bdc3c7'), spaceBefore=4, spaceAfter=8))
            continue

        # Blockquote / Callout
        if line.startswith('>'):
            callout_text = clean_text(line.lstrip('> ').strip())
            story.append(Paragraph(callout_text, callout_style))
            continue

        # Headings
        if line.startswith('# '):
            story.append(Paragraph(clean_text(line[2:]), title_style))
        elif line.startswith('## '):
            story.append(Paragraph(clean_text(line[3:]), h1_style))
        elif line.startswith('### '):
            story.append(Paragraph(clean_text(line[4:]), h2_style))
        # Bullet List
        elif line.startswith('* ') or line.startswith('- '):
            bullet_text = "• " + clean_text(line[2:])
            story.append(Paragraph(bullet_text, bullet_style))
        # Numbered List
        elif re.match(r'^\d+\.\s', line):
            num_text = clean_text(line)
            story.append(Paragraph(num_text, bullet_style))
        else:
            story.append(Paragraph(clean_text(line), body_style))

    doc.build(story)
    print(f"Successfully compiled PDF report to {REPORT_PDF}")

if __name__ == '__main__':
    build_pdf()
