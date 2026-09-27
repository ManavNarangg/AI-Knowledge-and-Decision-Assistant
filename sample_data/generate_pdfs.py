"""Generate the sample knowledge-base PDFs for two demo organizations.

    python sample_data/generate_pdfs.py

Creates:
  sample_data/northbridge_university/*.pdf   (Org A - a university)
  sample_data/acme_logistics/*.pdf           (Org B - a logistics company)
  sample_data/invalid/*                      (files that must be rejected by upload validation)
"""
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent


class PolicyPDF(FPDF):
    def __init__(self, org: str, title: str):
        super().__init__()
        self.org, self.doc_title = org, title
        self.set_auto_page_break(auto=True, margin=20)
        self.alias_nb_pages()

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(110)
        self.cell(0, 8, f"{self.org} - {self.doc_title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(130)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")


def build(org: str, title: str, pages: list[tuple[str, list[str]]], out: Path) -> None:
    pdf = PolicyPDF(org, title)
    for i, (heading, paragraphs) in enumerate(pages):
        pdf.add_page()
        if i == 0:
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(20)
            pdf.multi_cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30)
        pdf.multi_cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(40)
        for p in paragraphs:
            pdf.multi_cell(0, 6, p, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))
    print(f"  wrote {out.relative_to(ROOT)} ({len(pages)} pages)")


# --------------------------------------------------------------------------- Org A: Northbridge University
NB = "Northbridge University"

SCHOLARSHIP_RULES = [
    ("1. Overview of Scholarship Programs", [
        "Northbridge University offers three institutional scholarship programs: the Merit Scholarship, "
        "the Need-Based Grant and the Athletic (Sports) Scholarship. All programs are administered by the "
        "Office of Student Finance.",
        "The Merit Scholarship recognizes outstanding academic performance and is worth between $5,000 and "
        "$7,500 per academic year depending on the award tier. The Need-Based Grant provides up to $8,000 per "
        "academic year to students with demonstrated financial need. The Athletic Scholarship is awarded by the "
        "Athletics Department to varsity athletes and is worth between $3,000 and $10,000 per academic year.",
        "The application deadline for the Merit Scholarship and the Need-Based Grant is March 31 each year. "
        "Decisions are communicated by May 15. Awards are paid in two equal installments, one per semester.",
    ]),
    ("2. Merit Scholarship - Eligibility Requirements", [
        "To be eligible for the Merit Scholarship a student must meet ALL of the following requirements:",
        "(a) Enrollment: the student must be enrolled full-time, meaning at least 12 credits per semester.",
        "(b) Year of study: the student must be in year 2, 3 or 4 of an undergraduate program. First-year "
        "students are not eligible for the Merit Scholarship; they are automatically considered for the "
        "Entrance Award instead.",
        "(c) Academic standing: the student must have a cumulative GPA of at least 3.5 on a 4.0 scale.",
        "(d) Conduct: the student must have no record of academic misconduct.",
        "Meeting the requirements does not guarantee an award if funds are exhausted, but every eligible "
        "applicant who applies before the deadline is ranked by GPA.",
    ]),
    ("3. Merit Scholarship - Award Tiers", [
        "Eligible students are placed in exactly one award tier based on cumulative GPA:",
        "Gold tier: cumulative GPA of 3.85 or higher - $7,500 per academic year.",
        "Silver tier: cumulative GPA from 3.70 up to 3.84 - $6,000 per academic year.",
        "Bronze tier: cumulative GPA from 3.50 up to 3.69 - $5,000 per academic year.",
        "Tier placement is re-evaluated at every annual renewal.",
    ]),
    ("4. Need-Based Grant", [
        "The Need-Based Grant is available to students whose annual household income is below $60,000.",
        "Applicants must hold a cumulative GPA of at least 2.5 and may be enrolled full-time or part-time, "
        "with a minimum of 6 credits per semester.",
        "Applicants must submit the Financial Circumstances Form together with proof of household income. "
        "The grant amount (up to $8,000) is determined on a sliding scale according to household income.",
        "Unlike the Merit Scholarship, the Need-Based Grant is open to first-year students.",
    ]),
    ("5. Combining Awards (Stacking Rules)", [
        "A student may hold at most two institutional awards at the same time.",
        "A student who holds an Athletic (Sports) Scholarship may also receive the Merit Scholarship only if "
        "the Athletic Scholarship is worth less than $5,000 per academic year. Students whose Athletic "
        "Scholarship is $5,000 or more per year are not eligible to receive the Merit Scholarship in the same year.",
        "The combined value of all institutional awards may not exceed the student's annual tuition of $24,000. "
        "If it would, the Need-Based Grant is reduced first.",
    ]),
    ("6. Renewal, Revocation and Appeals", [
        "The Merit Scholarship is renewable each year provided the student maintains a cumulative GPA of at "
        "least 3.3 and remains enrolled full-time. A student whose GPA falls below 3.3 loses the scholarship "
        "for the following year but may re-apply once the GPA is 3.5 or higher.",
        "Any scholarship may be revoked for a finding of academic misconduct.",
        "Students may appeal a scholarship decision to the Scholarship Committee within 14 days of the decision. "
        "The appeal procedure is described in the Student Services Guide.",
    ]),
]

ACADEMIC_POLICIES = [
    ("1. Attendance", [
        "Students are expected to attend at least 75% of scheduled classes in every course.",
        "A student whose attendance in a course falls below 75% is barred from the final examination of that "
        "course unless a medical exemption is approved by the Dean of Students.",
        "Absences for official university athletic competitions are excused and do not count toward the limit.",
    ]),
    ("2. Grading, Late Work and Academic Probation", [
        "Late assignments lose 10% of the available marks for each calendar day they are late, for up to 5 days. "
        "Work submitted more than 5 days late receives a mark of zero.",
        "A student whose cumulative GPA falls below 2.0 is placed on academic probation. A student on probation "
        "for two consecutive semesters is referred for a dismissal review.",
        "Academic misconduct, including plagiarism, results in a zero for the assessment and a permanent "
        "record of misconduct.",
    ]),
    ("3. Course Load and Withdrawal", [
        "A full-time course load is 12 to 18 credits per semester. Part-time students take fewer than 12 credits.",
        "To take more than 18 credits in a semester (an overload), a student must have a cumulative GPA of at "
        "least 3.0 AND written approval from their academic advisor. The maximum load is 21 credits.",
        "Students may withdraw from a course up to the end of week 8 of the semester and receive a W grade, which "
        "does not affect GPA. Withdrawals after week 8 receive a WF grade, which counts as a failing grade.",
    ]),
]

STUDENT_SERVICES = [
    ("1. How to Apply for a Scholarship", [
        "Step 1: Log in to the Northbridge Student Portal and open the Financial Aid section.",
        "Step 2: Select the scholarship you are applying for and complete the online application form.",
        "Step 3: Upload the required documents. Merit Scholarship applicants must upload an official transcript, "
        "a personal statement of no more than 800 words and two academic references. Need-Based Grant applicants "
        "must upload the Financial Circumstances Form and proof of household income.",
        "Step 4: Submit the application before the March 31 deadline. Late applications are not considered.",
        "You will receive an email confirmation within 2 business days and a decision by May 15.",
    ]),
    ("2. Contacts and Office Hours", [
        "Office of Student Finance: Room 204, Administration Building. Email: finance@northbridge.edu. "
        "Phone: (555) 010-2040. Open Monday to Friday, 9:00 to 17:00.",
        "Academic Advising Center: Room 110, Library Building. Book appointments through the Student Portal.",
        "Dean of Students Office: Room 301, Administration Building. Handles medical exemptions and conduct matters.",
    ]),
    ("3. Appeals Procedure", [
        "To appeal a scholarship or academic decision, complete Form SF-12 (Decision Appeal) in the Student Portal "
        "within 14 days of the decision.",
        "Attach any supporting evidence, such as medical documentation or corrected transcripts.",
        "The Scholarship Committee meets every two weeks and will notify you of the outcome by email within "
        "21 days of receiving the appeal. The committee's decision is final.",
    ]),
]

# --------------------------------------------------------------------------- Org B: Acme Logistics
ACME = "Acme Logistics Ltd."

LEAVE_POLICY = [
    ("1. Annual Leave Entitlement", [
        "Full-time employees accrue paid annual leave based on their length of continuous service:",
        "Less than 2 years of service: 15 days per year.",
        "2 years up to (but not including) 5 years of service: 20 days per year.",
        "5 years of service or more: 25 days per year.",
        "Part-time employees receive a pro-rated entitlement based on contracted hours.",
        "Up to 5 unused days may be carried over into the next year and must be used by March 31. Any other "
        "unused days are forfeited and are not paid out.",
    ]),
    ("2. Sick Leave", [
        "Employees receive 10 days of paid sick leave per calendar year. Unused sick leave does not carry over.",
        "A medical certificate is required for any absence of more than 3 consecutive working days.",
        "Employees must notify their line manager before the start of their shift on each day of absence.",
    ]),
    ("3. Parental Leave", [
        "Employees are eligible for paid parental leave after 12 months of continuous employment.",
        "The primary caregiver receives 16 weeks of leave at full pay. The secondary caregiver receives 4 weeks "
        "of leave at full pay. Both may be taken any time within 12 months of the birth or adoption.",
        "Employees must give at least 8 weeks' written notice to HR before the intended start of parental leave.",
        "Employees with less than 12 months of service may take up to 12 weeks of unpaid parental leave.",
    ]),
]

REMOTE_WORK = [
    ("1. Hybrid Remote Work - Eligibility", [
        "Acme Logistics supports hybrid working of up to 3 days per week from home for eligible employees.",
        "An employee is eligible for hybrid remote work only if ALL of the following conditions are met:",
        "(1) The employee has completed the probation period, which is 6 months of service.",
        "(2) The employee's most recent performance rating is 3 or higher on the 5-point scale.",
        "(3) The employee's role is not designated as on-site essential. The on-site essential roles are "
        "warehouse operations, fleet maintenance and dispatch.",
        "(4) The employee's line manager has approved the request.",
    ]),
    ("2. Fully Remote Work", [
        "Fully remote work (5 days per week) is an exception and requires director approval.",
        "To be considered for fully remote work, an employee must have at least 2 years of service, a most "
        "recent performance rating of 4 or higher, and work in the Technology, Finance or Human Resources department.",
        "Fully remote employees must attend the office for the quarterly planning day.",
        "Approved remote workers (hybrid or fully remote) receive a one-time home office equipment stipend of $500.",
    ]),
    ("3. How to Request Remote Work", [
        "Submit the Remote Work Request form in the HR Portal, selecting either hybrid or fully remote.",
        "Your line manager reviews the request within 10 business days. Fully remote requests are then "
        "forwarded to your director.",
        "Approved arrangements are reviewed every 6 months and may be withdrawn if performance falls below a "
        "rating of 3 or business needs change.",
    ]),
]

EXPENSES = [
    ("1. Reimbursement Limits", [
        "Meals: up to $75 per day for domestic travel and up to $100 per day for international travel.",
        "Hotels: up to $200 per night for domestic travel and up to $300 per night for international travel.",
        "Personal vehicle use is reimbursed at $0.60 per mile.",
        "Alcohol, personal entertainment, traffic fines and upgrades to business class are never reimbursable.",
    ]),
    ("2. Receipts and Approval", [
        "An itemized receipt is required for every expense over $25. Expenses over $25 without a receipt are "
        "not reimbursed.",
        "Claims must be submitted within 30 days of the expense date. Claims submitted later are rejected.",
        "Claims below $1,000 are approved by the line manager. Claims of $1,000 or more also require director approval.",
    ]),
    ("3. How to Submit an Expense Claim", [
        "Step 1: Log in to ExpensePoint, the Acme expense portal.",
        "Step 2: Create a new claim, choose the expense category and enter the amount and date.",
        "Step 3: Photograph or upload each receipt and attach it to the matching line.",
        "Step 4: Submit the claim for approval. Approved claims are paid with the next payroll run, within "
        "10 business days of approval.",
        "Questions about expenses can be sent to expenses@acmelogistics.example.",
    ]),
]


SCANNED_NOTICE = [
    "NORTHBRIDGE UNIVERSITY - CAMPUS PARKING NOTICE",
    "",
    "1. Student parking permits cost $120 per semester.",
    "2. Permits are sold online through the Student Portal from August 15.",
    "3. Parking without a permit results in a $40 fine.",
    "4. Students with a disability permit may park in any blue bay free of charge.",
    "5. Overnight parking is only allowed in Lot C.",
]


def build_scanned(out: Path) -> None:
    """A PDF whose page is a picture of text (no text layer), like a scan or 'Print to PDF' of an image."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("L", (1240, 1754), color=255)           # A4 at 150 DPI
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 34)
        title = ImageFont.truetype("arialbd.ttf", 40)
    except OSError:
        font = title = ImageFont.load_default(size=34)
    y = 150
    for i, line in enumerate(SCANNED_NOTICE):
        draw.text((110, y), line, fill=20, font=title if i == 0 else font)
        y += 80 if i == 0 else 60
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    image.save(png)
    pdf = FPDF(format="A4")
    pdf.add_page()
    pdf.image(str(png), x=0, y=0, w=210, h=297)
    pdf.output(str(out))
    png.unlink()
    print(f"  wrote {out.relative_to(ROOT)} (1 image-only page)")


def main() -> None:
    print("Northbridge University")
    build(NB, "Scholarship Rules 2026", SCHOLARSHIP_RULES, ROOT / "northbridge_university" / "Scholarship_Rules.pdf")
    build(NB, "Academic Policies", ACADEMIC_POLICIES, ROOT / "northbridge_university" / "Academic_Policies.pdf")
    build(NB, "Student Services Guide", STUDENT_SERVICES, ROOT / "northbridge_university" / "Student_Services_Guide.pdf")

    print("Acme Logistics")
    build(ACME, "Leave Policy", LEAVE_POLICY, ROOT / "acme_logistics" / "Leave_Policy.pdf")
    build(ACME, "Remote Work Policy", REMOTE_WORK, ROOT / "acme_logistics" / "Remote_Work_Policy.pdf")
    build(ACME, "Expense Reimbursement Policy", EXPENSES, ROOT / "acme_logistics" / "Expense_Reimbursement_Policy.pdf")

    print("Scanned PDF (image-only pages, exercises OCR)")
    build_scanned(ROOT / "scanned" / "Parking_Notice_scanned.pdf")

    print("Invalid files (for error-handling demos)")
    invalid = ROOT / "invalid"
    invalid.mkdir(exist_ok=True)
    (invalid / "not_really_a.pdf").write_text("This is a text file pretending to be a PDF.\n")
    long_doc = [(f"Section {i}", ["Filler content for page-limit testing."]) for i in range(1, 26)]
    build("Test", "Too Many Pages", long_doc, invalid / "too_many_pages_25.pdf")
    blank = FPDF()
    blank.add_page()
    blank.output(str(invalid / "scanned_no_text.pdf"))
    print("  wrote invalid/not_really_a.pdf, invalid/scanned_no_text.pdf")


if __name__ == "__main__":
    main()
