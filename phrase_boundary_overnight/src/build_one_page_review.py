"""One-page Chinese project review from verified local results; no training."""
from pathlib import Path
import csv
import json
import hashlib
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from pypdf import PdfReader

PROJECT=Path(__file__).resolve().parents[2]
REPORTS=PROJECT/'phrase_boundary_overnight/reports'
OUT=PROJECT/'output/pdf/乐句划分研究_一页Review.pdf'
QA=PROJECT/'tmp/pdfs/phrase_review'


def pick(name, **filters):
    with (REPORTS/name/'means.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if all(r[k]==v for k,v in filters.items())]
    assert len(rows)==1,(name,filters)
    return rows[0]


def main():
    OUT.parent.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
    pdfmetrics.registerFont(TTFont('ReviewCN','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('ReviewCN-Bold','C:/Windows/Fonts/msyhbd.ttc',subfontIndex=0))
    pdfmetrics.registerFontFamily('ReviewCN',normal='ReviewCN',bold='ReviewCN-Bold')
    current=float(pick('gru_four_seed_summary',kind='E',policy='M10',subset='all_four')['macro_f1_tol1'])*100
    ref=float(pick('gru_four_seed_summary',kind='C3',policy='M10',subset='all_four')['macro_f1_tol1'])*100
    audio=float(pick('external_structure_probe',kind='M')['f1_tol1'])*100
    audio_shuffle=float(pick('external_structure_probe',kind='MD')['f1_tol1'])*100
    baseline_record=json.loads((REPORTS/'teacher_report_20260911/brief_edit_record.json').read_text(encoding='utf-8-sig'))
    historical=float(baseline_record['historical_baseline_requested_by_user'])
    delta=round(current,2)-historical
    relative=delta/historical*100
    assert historical==35.06 and not baseline_record['baseline_verified_as_transformer']
    assert round(current,2)==64.14 and round(audio,2)==24.66
    width,height=A4;left=44;usable=width-88;y=height-43
    c=canvas.Canvas(str(OUT),pagesize=A4);c.setTitle('乐句划分研究 - 一页Review');c.setAuthor('')
    ink=HexColor('#172033');muted=HexColor('#526071');accent=HexColor('#205b77')
    body=ParagraphStyle('body',fontName='ReviewCN',fontSize=10.5,leading=16.5,textColor=ink,wordWrap='CJK')
    cell=ParagraphStyle('cell',parent=body,fontSize=9.9,leading=15)
    note=ParagraphStyle('note',parent=body,fontSize=8.6,leading=13,textColor=muted)

    def para(text,style=body,gap=5):
        nonlocal y
        p=Paragraph(text,style);_,h=p.wrap(usable,800)
        assert y-h>=43,('Page overflow',text[:30],y,h)
        p.drawOn(c,left,y-h);y-=h+gap

    def heading(number,title):
        nonlocal y
        y-=17;c.setFillColor(accent);c.setFont('ReviewCN-Bold',12)
        c.drawString(left,y,f'{number}  {title}');y-=17

    c.setFillColor(ink);c.setFont('ReviewCN-Bold',20);c.drawString(left,y,'乐句划分研究：进展与问题');y-=25
    c.setFillColor(muted);c.setFont('ReviewCN',9);c.drawString(left,y,'全程回顾  |  2026年9月14日  |  从早期曲线Transformer到当前融合模型');y-=13
    c.setStrokeColor(HexColor('#cbd5df'));c.line(left,y,width-left,y);y-=13
    para(f'<b>历史分数变化：{historical:.2f}% → {current:.2f}%</b>，增加<b>{delta:.2f}个百分点</b>，相对增加<b>{relative:.2f}%</b>。这是全程数值变化，不是同条件下单个算法的收益。',gap=4)
    para('35.06%沿用此前指定的历史值，尚未对应原始结果表；已核验早期输入含9维速度/力度曲线，并非仅一条tempo curve。期间由句末改为句首，标签与评价也有修订。',style=note,gap=0)

    heading('01','主要问题在哪里')
    para('<b>有变化，不一定是新乐句。</b>已核对的谱例中，低声部先开句仍会漏报，句尾停顿又可能被当作句首。模型已能利用重复关系，但尚未证明学会了“前句结束、后句重新开始”。')
    para('<b>录音不少，独立作品仍少。</b>主数据43首玛祖卡，当前反复调试的是19首；新增21个外部乐章也仅部分标注。64.14%是开发成绩，不能代表全新曲目的表现。')
    para('<b>真实使用还有对齐这一关。</b>外部音频实验用了参考拍时间。旧系统的自动对齐误差会带来多少断句损失，尚未验证。',gap=0)

    heading('02','全程关键尝试与结果')
    data=[
        [Paragraph('<b>主要改动</b>',cell),Paragraph('<b>该阶段F1对照</b>',cell),Paragraph('<b>说明了什么</b>',cell)],
        [Paragraph('加入谱面，比较局部CNN',cell),Paragraph('小CNN：9维32.06%<br/>→ 25维40.21%',cell),Paragraph('谱面有帮助；同25维下CNN+Transformer为38.99%，也胜纯Transformer的30.96%。',cell)],
        [Paragraph('加入多尺度速度统计',cell),Paragraph('38.24% → 42.02%<br/>+3.77个百分点',cell),Paragraph('较长时间范围有用；这里的“能量”来自速度曲线，不是声波能量。',cell)],
        [Paragraph('保留音符顺序与重复关系',cell),Paragraph('44.03% → 53.84%<br/><b>+9.80个百分点</b>',cell),Paragraph('较强的正面证据：让模型看到前后片段怎样对应，而非只看均值。',cell)],
        [Paragraph('CNN从1层增至3层',cell),Paragraph('53.84% → 55.09%<br/>+1.25个百分点',cell),Paragraph('只有小幅收益；后加局部Transformer虽涨容差F1，精确定位反而下降。',cell)],
        [Paragraph('按重复与间距筛选边界',cell),Paragraph(f'58.53% → {ref:.2f}%<br/>+4.05个百分点',cell),Paragraph('主要减少误报；模型权重没变，不能说全靠学到新特征。',cell)],
        [Paragraph('CNN与双向GRU融合',cell),Paragraph(f'{ref:.2f}% → <b>{current:.2f}%</b><br/>+{current-ref:.2f}个百分点',cell),Paragraph('当前保留方案；两个CNN也接近该成绩，未证明GRU有独有优势。',cell)],
        [Paragraph('外部音频与频谱探索',cell),Paragraph(f'音频前后变化 {audio:.2f}%<br/>打乱时间 {audio_shuffle:.2f}%',cell),Paragraph('音频有信号，但直接加频谱未稳定改善；不能与主任务横比。',cell)]
    ]
    table=Table(data,colWidths=[142,160,usable-302])
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#eef3f7')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.6,HexColor('#b8c7d1')),('LINEBELOW',(0,1),(-1,-1),.35,HexColor('#dde4e9'))]))
    _,h=table.wrap(usable,800);assert y-h>115;table.drawOn(c,left,y-h);y-=h+7
    para('表内每行是该阶段自己的对照，不同阶段不可相加。另已试句首/句尾联合、分轴卷积、迁移学习、和声辅助及树模型，尚未得到可稳定替换当前方案的结果。',style=note,gap=3)
    para('指标：F1综合误报和漏报，按±1个四分音符拍一对一匹配、按作品汇总，并非逐拍正确率。64.14%来自4次初始化×2个开发折，融合提升仍不能排除随机波动。',style=note,gap=0)

    heading('03','当前方案')
    para('58维谱面、速度/力度及其派生量、重复关系 → 三层CNN与单层双向GRU分别预测 → 概率各占50% → 重复/间距筛选 → 每拍句首位置。主方案不含原始音频；音乐理解与跨曲泛化仍是阶段性结论。',gap=0)
    assert y>=51,y
    c.setStrokeColor(HexColor('#d9e1e7'));c.line(left,34,width-left,34)
    c.setFillColor(muted);c.setFont('ReviewCN',7.5);c.drawString(left,21,'依据：局部CNN、切片/速度统计、保序重复、深度、边界筛选、四种子融合及外部音频报告。')
    c.drawRightString(width-left,21,'1 / 1');c.showPage();c.save()
    reader=PdfReader(OUT);assert len(reader.pages)==1
    text=reader.pages[0].extract_text();required=['主要问题在哪里','全程关键尝试与结果','当前方案','35.06%','64.14%','29.08','82.94%','53.84%','42.02%']
    assert all(t in text for t in required),[t for t in required if t not in text]
    assert '下一步计划' not in text
    (QA/'extracted.txt').write_text(text,encoding='utf-8')
    audit=dict(pages=1,sections=required[:3],font_pt=10.5,body_bottom_pt=y,required_text_verified=True,
        historical_score=historical,historical_baseline_verified=False,delta_pp=round(delta,2),relative_percent=round(relative,2),
        pdf_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),visual_review='pending')
    (QA/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False));print(OUT)


if __name__=='__main__':main()
