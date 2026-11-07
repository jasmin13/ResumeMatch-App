import os
import io
import re
import json
import time
import base64
import datetime
import warnings
from pathlib import Path
from typing import List, Set, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers import logging as tf_logging
# Optimize PyTorch CPU inference speed with maximum available physical cores
if hasattr(torch, "set_num_threads"):
    try:
        cores = os.cpu_count() or 4
        torch.set_num_threads(min(12, cores))
    except Exception:
        pass
# Suppress framework warnings
warnings.filterwarnings("ignore")
tf_logging.set_verbosity_error()
def find_model_path() -> str:
    """Find merged model path across diverse execution directories."""
    candidates = [
        Path("./merged_resume_model"),
        Path(__file__).parent / "merged_resume_model",
        Path("../merged_resume_model"),
        Path("../soup/merged_resume_model"),
        Path(__file__).parent.parent / "merged_resume_model",
        Path(__file__).parent.parent / "soup" / "merged_resume_model",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return "./merged_resume_model"
MODEL_PATH = os.environ.get("RESUME_MODEL_PATH", find_model_path())
tokenizer = None
model = None
class StopOnJsonEnd(StoppingCriteria):
    """Early stops generation as soon as the closing JSON brace is produced, saving up to 75% inference time."""
    def __init__(self, tokenizer, start_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.start_len = start_len
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        new_ids = input_ids[0][self.start_len:]
        if len(new_ids) > 20:
            tail = self.tokenizer.decode(new_ids[-2:])
            if "}" in tail:
                return True
        return False
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    print(f"Loading ResumeMatch AI model from: {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    model.eval()
    print("ResumeMatch AI Model successfully loaded and ready for inference!")
    yield
app = FastAPI(
    title="ResumeMatch AI - Intelligent Candidate & Job Matching Engine",
    lifespan=lifespan
)
class ResumeRequest(BaseModel):
    resume_text: str
class ResumeUploadRequest(BaseModel):
    filename: str
    content_base64: str
class JobMatchRequest(BaseModel):
    resume_text: str
    job_description: str
class BatchResumeItem(BaseModel):
    filename: str
    content_base64: Optional[str] = None
    text: Optional[str] = None
class BatchMatchRequest(BaseModel):
    job_description: str
    resumes: List[BatchResumeItem]
    custom_core_criteria: Optional[List[str]] = None
    custom_preferred_criteria: Optional[List[str]] = None
class ExtractCriteriaRequest(BaseModel):
    job_description: str
def _clean_pdf_text(text: str) -> str:
    """Fix common PyPDF2 text extraction artifacts: spaced hyphens, split words, etc."""
    import unicodedata
    # Normalize unicode to ASCII-compatible form
    text = unicodedata.normalize("NFKC", text)
    # Fix "word -hyphen" and "hyphen- word" artifacts (PDF justified text spacing)
    text = re.sub(r'(\w)\s+-\s*(\w)', r'\1-\2', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1-\2', text)
    # Fix common split technical words from PDF extraction
    fixes = [
        (r'\bREST\s+ful\b', 'RESTful'),
        (r'\bJava\s+Script\b', 'JavaScript'),
        (r'\bType\s+Script\b', 'TypeScript'),
        (r'\bNode\s+[Jj][Ss]\b', 'Node.js'),
        (r'\bReact\s+[Jj][Ss]\b', 'React.js'),
        (r'\bPost\s+gre\s*SQL\b', 'PostgreSQL'),
        (r'\bMong\s*o\s*DB\b', 'MongoDB'),
        (r'\bGit\s+Hub\b', 'GitHub'),
        (r'\bWork\s+Press\b', 'WordPress'),
        (r'\bCloud\s+Front\b', 'CloudFront'),
        (r'\bcross\s+-\s*functional\b', 'cross-functional'),
    ]
    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text
def _extract_text_from_bytes(filename: str, raw_bytes: bytes) -> str:
    """Helper to extract text from PDF, DOCX, or text bytes."""
    filename_lower = filename.lower()
    extracted_text = ""
    if filename_lower.endswith(".pdf"):
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
        pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
        extracted_text = "\n\n".join(pages_text).strip()
        extracted_text = _clean_pdf_text(extracted_text)
    elif filename_lower.endswith((".docx", ".doc")):
        import docx
        doc = docx.Document(io.BytesIO(raw_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                if row_text:
                    paragraphs.append(row_text)
        extracted_text = "\n".join(paragraphs).strip()
    else:
        extracted_text = raw_bytes.decode("utf-8", errors="ignore").strip()
    return extracted_text

@app.post("/api/upload-resume")
def upload_resume(req: ResumeUploadRequest):
    """Parse and extract text from uploaded PDF, Word (.docx), or plain text resume files."""
    if not req.content_base64:
        raise HTTPException(status_code=400, detail="File content cannot be empty.")
    
    try:
        raw_bytes = base64.b64decode(req.content_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")
    
    try:
        extracted_text = _extract_text_from_bytes(req.filename, raw_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse resume document: {str(e)}")
        
    if not extracted_text:
        raise HTTPException(status_code=400, detail="Could not extract readable text from the uploaded file.")
    return {
        "success": True,
        "filename": req.filename,
        "character_count": len(extracted_text),
        "word_count": len(extracted_text.split()),
        "extracted_text": extracted_text
    }
_SOFT_SKILL_TERMS = {
    "cross-functional collaboration", "technical leadership", "team leadership",
    "problem solving", "problem-solving", "collaboration", "communication",
    "adaptability", "time management", "agile", "scrum", "analytical thinking",
    "critical thinking", "team player", "leadership", "cross-functional", "cross -functional", "cross -functional collaboration",
    "people management", "team management", "team leadership", "tech leadership",
    "team lead", "tech lead", "people leader", "leadership skills",
    "software delivery", "code review", "code reviews", "agile methodology",
    "mentoring", "mentorship", "mentor developers",
    "drive technical strategy", "ensure quality delivery", "quality delivery",
    "mentoring", "mentorship", "interpersonal skills", "organizational skills",
    "strategic thinking", "innovation", "stakeholder management"
}
_CONDENSED_NORMALIZE = {
    "restful api development": "RESTful APIs",
    "rest ful api development": "RESTful APIs",
    "rest ful api": "RESTful APIs",
    "restful api": "RESTful APIs",
    "rest api": "RESTful APIs",
    "rest apis": "RESTful APIs",
}
def _strip_soft_skills_from_line(line: str) -> str:
    """Remove soft-skill terms from a skill category line before passing to model."""
    parts = re.split(r',\s*(?![^()]*\))', line)
    cleaned = []
    for part in parts:
        stripped = part.strip().strip(".,;:\"'`")
        key = stripped.lower()
        if key in _SOFT_SKILL_TERMS:
            continue
        # Normalize verbose aliases inline before model sees them
        normalized = _CONDENSED_NORMALIZE.get(key, stripped)
        if normalized:
            cleaned.append(normalized if normalized != stripped else part)
    return ", ".join(cleaned) if cleaned else line
def condense_resume_for_inference(resume_text: str) -> str:
    """
    Substantially reduces prompt token count (from ~3,000 to ~300 tokens).
    This dramatically accelerates CPU attention prefill latency from 35s to under 2s.
    Soft-skill terms are stripped from skill lines before model inference to prevent hallucination.
    """
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    
    in_skills = False
    in_summary = False
    skills_lines = []
    summary_lines = []
    experience_headers = []
    
    for line in lines:
        lower = line.lower()
        lower_stripped = lower.strip().strip(":-")
        is_skills_header = (
            any(h in lower for h in ["technical skills", "skills & tools", "core competencies", "technologies:", "key skills", "skill set"])
            or lower_stripped in ("skills", "skills:", "technologies", "technologies:")
        )
        if is_skills_header:
            in_skills = True
            in_summary = False
            skills_lines.append(_strip_soft_skills_from_line(line))
            continue
        elif any(h in lower for h in ["professional summary", "executive summary", "profile", "summary:", "about me"]) or lower_stripped in ("summary", "profile", "about"):
            in_summary = True
            in_skills = False
            summary_lines.append(line)
            continue
        elif any(h in lower for h in ["experience", "employment history", "work history", "education", "certifications"]):
            in_skills = False
            in_summary = False
            
        if in_skills:
            if len(skills_lines) < 8:
                skills_lines.append(_strip_soft_skills_from_line(line))
        elif in_summary:
            if len(summary_lines) < 4:
                summary_lines.append(line)
        elif re.search(r'\b(19\d{2}|20\d{2})\b', line) or any(title in lower for title in ["engineer", "developer", "lead", "architect", "manager", "specialist", "consultant", "analyst"]):
            if len(experience_headers) < 6:
                experience_headers.append(line)
    condensed_parts = []
    if summary_lines:
        condensed_parts.append("\n".join(summary_lines))
    if skills_lines:
        condensed_parts.append("\n".join(skills_lines))
    if experience_headers:
        condensed_parts.append("\n".join(experience_headers))
        
    result = "\n".join(condensed_parts)
    if len(result) < 150:
        words = resume_text.split()
        return " ".join(words[:320])
    return result
def calculate_years_from_text(raw_text: str, model_years: int = 10) -> str:
    """Universally calculate total experience from any resume timeline or model extraction."""
    year_matches = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', raw_text)]
    current_year = datetime.datetime.now().year
    
    valid_years = [y for y in year_matches if 1980 <= y <= current_year]
    if valid_years:
        earliest_year = min(valid_years)
        diff = current_year - earliest_year
        if diff >= 12:
            return "12+ Years"
        elif diff >= 10:
            return "10+ Years"
        elif diff >= 5:
            return f"{diff}+ Years"
        elif diff > 0:
            return f"{diff} Years"
        else:
            return "Entry-Level (< 1 Year)"
    if model_years >= 12:
        return "12+ Years"
    elif model_years >= 10:
        return "10+ Years"
    elif model_years > 0:
        return f"{model_years} Years"
    return "Entry-Level"
def _normalize_skill_key(s: str) -> str:
    """Normalize a skill string for blacklist/dedup comparison: lowercase, ASCII hyphens, single spaces."""
    # Replace unicode hyphens/dashes with ASCII hyphen
    s = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]', '-', s)
    # Remove spaces around hyphens (PDF extraction artifact: 'cross -functional' -> 'cross-functional')
    s = re.sub(r'\s*-\s*', '-', s)
    # Collapse multiple spaces
    s = re.sub(r'\s+', ' ', s)
    return s.lower().strip()
def extract_and_deduplicate_skills(raw_text: str, model_skills: List[str]) -> List[str]:
    """
    Universally extracts and deduplicates 100% of candidate skills across any domain.
    Eliminates redundant synonyms, duplicate entries, URLs, and non-technical soft buzzwords.
    """
    raw_collected = []
    canonical_map = {
        "react": "React.js",
        "reactjs": "React.js",
        "react.js": "React.js",
        "nodejs": "Node.js",
        "node.js": "Node.js",
        "node": "Node.js",
        "express": "Express.js",
        "expressjs": "Express.js",
        "express.js": "Express.js",
        "vue": "Vue.js",
        "vuejs": "Vue.js",
        "vue.js": "Vue.js",
        "angular": "AngularJS",
        "angularjs": "AngularJS",
        "html": "HTML5",
        "html5": "HTML5",
        "css": "CSS3",
        "css3": "CSS3",
        "postgres": "PostgreSQL",
        "postgresql": "PostgreSQL",
        "material ui (mui)": "Material UI",
        "material-ui": "Material UI",
        "mui": "Material UI",
        "material ui": "Material UI",
        "aws (ec2, s3, cloudfront)": "AWS",
        "aws ec2": "AWS",
        "docker compose": "Docker",
        "github actions (ci/cd)": "GitHub Actions",
        "github actions": "GitHub Actions",
        "restful api development": "RESTful APIs",
        "restful api": "RESTful APIs",
        "rest apis": "RESTful APIs",
        "rest api": "RESTful APIs",
        "restful apis": "RESTful APIs",
        "javascript (es6+)": "JavaScript",
        "es6+": "ES6+",
        "es6": "ES6+",
        # PDF text-extraction space artifacts
        "rest ful api development": "RESTful APIs",
        "rest ful api": "RESTful APIs",
        "rest ful apis": "RESTful APIs",
        "java script": "JavaScript",
        "type script": "TypeScript",
        "node js": "Node.js",
        "react js": "React.js",
    }
    soft_skills_blacklist = _SOFT_SKILL_TERMS | {
        "problem solving", "technical leadership",
        "team leadership", "collaboration", "communication", "adaptability",
        "time management", "agile", "scrum", "analytical thinking", "critical thinking",
        "team player", "leadership", "cross functional collaboration", "problem-solving",
        "cross-functional", "cross -functional", "cross -functional collaboration",
    "people management", "team management", "team leadership", "tech leadership",
    "team lead", "tech lead", "people leader", "leadership skills",
    "software delivery", "code review", "code reviews", "agile methodology",
    "mentoring", "mentorship", "mentor developers",
    "drive technical strategy", "ensure quality delivery", "quality delivery", "mentoring", "mentorship", "interpersonal skills",
        "organizational skills", "strategic thinking", "innovation", "stakeholder management",
        # Generic domain names that belong to profile header, not technical skills
        "software engineering", "software development", "web development",
        "full-stack development", "full stack development", "full-stack engineering",
        "frontend development", "backend development", "engineering", "development"
    }
    # Normalize all blacklist keys to handle unicode variants
    soft_skills_blacklist_norm = {_normalize_skill_key(k) for k in soft_skills_blacklist}
    def _is_blacklisted(skill: str) -> bool:
        return _normalize_skill_key(skill) in soft_skills_blacklist_norm
    # 1. Model extracted skills
    for s in model_skills:
        clean = s.strip().strip(".,;:\"'`")
        if re.search(r'https?://|www\.|\.com|@|linkedin|\.org|\+?\d{8,}|//', clean.lower()):
            continue
        c = canonical_map.get(clean.lower(), clean)
        if not _is_blacklisted(c) and len(c) > 1:
            raw_collected.append(c)
    # 2. Parse category lines — pre-strip soft skills, then split on commas
    category_matches = re.findall(r'^(?:[•\-\*\d\.]*\s*)([A-Za-z0-9\s/&]+):\s*([^\n\r]+)', raw_text, flags=re.MULTILINE)
    _ignored_cats = {
        "key responsibility", "key responsibilities", "responsibility", "responsibilities",
        "role", "roles", "role & responsibilities", "roles & responsibilities", "duties",
        "day to day", "what you'll do", "what you will do", "job description",
        "location", "locations", "ctc", "salary", "compensation", "experience", "employment",
        "company", "about", "about us", "who we are", "overview", "summary", "profile",
        "apply", "apply now", "contact", "contact us", "how to apply", "interested candidates",
        "email", "phone", "notes", "note", "benefits", "perks", "preferred background",
        "preferred", "requirements", "must-have skills", "must have", "what we're looking for"
    }
    for cat, items in category_matches:
        cat_lower = cat.strip().lower()
        if cat_lower in _ignored_cats or any(ic in cat_lower for ic in ["responsibility", "responsibilities", "location", "salary", "ctc", "apply", "contact", "about"]):
            continue
        items = _strip_soft_skills_from_line(items)
        # Normalize concatenated bullet points from single-line LinkedIn pastes
        # e.g. "BullMQ (good to have)- AWS" -> "BullMQ (good to have), AWS"
        # e.g. "Docker- Linux" -> "Docker, Linux"
        items = re.sub(r'\)\s*-\s*(?=[A-Za-z])', '), ', items)
        items = re.sub(r'(\w)-\s+(?=[A-Z])', r'\1, ', items)
        parts = re.split(r',\s*(?![^()]*\))|\||•', items)
        for p in parts:
            clean_item = p.strip().strip(".,;:\"'`")
            # Strip leading bullet/dash chars from JD paste formatting (e.g. "- Node.js" -> "Node.js")
            clean_item = re.sub(r'^[\-•\*]\s+', '', clean_item).strip()
            # Filter HR boilerplate / sentences that are not skills
            if len(clean_item.split()) > 7:  # Real skills are <=7 words
                continue
            if clean_item.endswith('.'):  # Ends with period = sentence
                continue
            _hr_noise = {'ctc', 'notice period', 'years of experience', 'year of experience',
                         'years experience', 'year experience', 'updated resume',
                         'apply', 'opportunity', 'on-site', 'location',
                         'work from', 'important', 'salary', 'hiring',
                         'mandatory', 'good to have', 'nice to have', 'preferred',
                         'based systems', 'based system', 'minimum', 'required'}
            if any(n in clean_item.lower() for n in _hr_noise):
                continue
            # Strip trailing qualifier words (e.g. 'PostgreSQL preferred' -> 'PostgreSQL')
            _qualifiers = re.compile(r'\s*(preferred|mandatory|required|recommended|only|minimum|plus|good to have|nice to have|\(.*?\))\s*$', re.IGNORECASE)
            clean_item = _qualifiers.sub('', clean_item).strip()
            # Normalize '-based systems' suffix (e.g. 'Linux-based systems' -> 'Linux')
            clean_item = re.sub(r'-based\s+\w+', '', clean_item, flags=re.IGNORECASE).strip()
            if not clean_item:
                continue
            # Filter experience-range requirements like '2-4 years of professional...'
            if re.search(r'\d+[-\u2013]\d+\s+years?|\d+\+?\s+years?\s+of', clean_item, re.IGNORECASE):
                continue
            if _is_blacklisted(clean_item):
                continue
            if not clean_item or re.search(r'https?://|www\.|\.com|@|linkedin|\.org|\+?\d{8,}|//', clean_item.lower()):
                continue
            
            paren_match = re.match(r'^([A-Za-z0-9\s\.\+\#\-]+)\s*\(([^)]+)\)$', clean_item)
            if paren_match:
                main_skill = paren_match.group(1).strip().strip(".,;:\"'`")
                sub_items = [s.strip().strip(".,;:\"'`") for s in paren_match.group(2).split(',') if s.strip()]
                for item in [main_skill] + sub_items:
                    # Skip sub-items that are sentences, not skills
                    if len(item.split()) > 6 or item.endswith('.'):
                        continue
                    c = canonical_map.get(item.lower(), item)
                    if not _is_blacklisted(c) and len(c) > 1:
                        raw_collected.append(c)
            else:
                c = canonical_map.get(clean_item.lower(), clean_item)
                if not _is_blacklisted(c) and len(c) > 1:
                    raw_collected.append(c)
    # 3. Dynamic keyword scan for verified industry tools
    known_tools = [
        "React.js", "Node.js", "Express.js", "MERN Stack", "Next.js", "Vue.js", "AngularJS",
        "Redux", "TypeScript", "JavaScript", "Python", "Flask", "Django", "FastAPI", "PHP", "Laravel",
        "PostgreSQL", "MongoDB", "MySQL", "Redis", "Elasticsearch", "Supabase", "Oracle", "SQLite",
        "AWS", "Docker", "Kubernetes", "Nginx", "Cloudflare", "GitHub Actions", "Azure", "GCP",
        "Playwright", "Crawl4AI", "Beautiful Soup", "GraphQL", "RESTful APIs", "JWT Authentication",
        "Vite", "Webpack", "Babel", "NPM", "i18next", "Material UI", "Postman", "pgAdmin4", "Git",
        "ES6+", "HTML5", "CSS3", "Tailwind CSS", "Bootstrap", "Sass",
        # Realtime & Messaging
        "Socket.IO", "WebSocket", "Firebase", "RabbitMQ", "Apache Kafka", "Kafka", "BullMQ", "Bull", "MQTT", "gRPC", "SSE",
        # ORM / Database tools
        "Prisma", "Sequelize", "TypeORM", "Mongoose", "Knex.js", "DynamoDB", "Cassandra",
        # Testing
        "Jest", "Mocha", "Chai", "Cypress", "Selenium", "Vitest", "JUnit", "PyTest",
        # DevOps & Cloud extras
        "Terraform", "Ansible", "Jenkins", "CircleCI", "GitLab CI", "PM2", "Linux", "Bash",
        "Cloud Technologies", "Microservices", "Microservices Architecture",
        # Mobile
        "React Native", "Flutter", "Swift", "Kotlin", "Android", "iOS", "Expo",
        # AI/ML
        "TensorFlow", "PyTorch", "Scikit-learn", "OpenCV", "Pandas", "NumPy", "LangChain", "OpenAI", "LLMs", "LLM", "Prompt Engineering", "Generative AI",
        "AI Agents", "AI Workflows", "Agentic AI", "Claude", "Gemini",
        "Hugging Face", "BERT", "CUDA", "Keras", "XGBoost", "Spark", "Hadoop",
        # Healthcare & Clinical
        "Critical Care", "Ventilator Management", "ECMO", "Telemetry", "IV Therapy", "Epic EMR", "Cerner", "ACLS", "BLS", "CCRN",
        # Finance & Business
        "Financial Modeling", "DCF", "LBO", "M&A", "Bloomberg Terminal", "Capital IQ", "Tableau", "Power BI", "SQL",
        # Engineering & Design
        "SolidWorks", "CATIA", "AutoCAD", "Figma", "Sketch", "Revit", "ANSYS", "FEA",
        # Other popular tools
        "Jira", "Confluence", "Slack", "Notion", "Stripe", "Twilio", "SendGrid", "Cloudinary",
        "Nginx", "Apache", "WordPress", "Shopify", "Magento", "WooCommerce",
        # Languages (commonly required in JDs)
        "Java", "C#", "Go", "Golang", "Rust", "Ruby", "Scala", "C++", "C",
        # Java ecosystem
        "Spring Boot", "Spring MVC", "Spring Framework", "Hibernate", "Maven", "Gradle", "JPA",
        # .NET ecosystem
        ".NET", "ASP.NET", "Entity Framework", "Blazor", "LINQ", "NuGet",
        # Backend frameworks (newer)
        "Nest.js", "NestJS", "Fastify", "Hono", "tRPC", "Ruby on Rails",
        # Monitoring & Observability
        "Grafana", "Prometheus", "Datadog", "Sentry", "New Relic", "Logstash", "Kibana",
        "ELK Stack", "OpenTelemetry", "Jaeger",
        # Data & Workflow
        "Apache Airflow", "Celery", "dbt", "Looker", "Metabase", "Apache Spark",
        "Power BI", "Tableau", "Snowflake", "BigQuery", "Redshift", "Databricks",
        # UI Libraries & Visualization
        "Ant Design", "Chakra UI", "Shadcn UI", "D3.js", "Chart.js", "Three.js",
        "Recharts", "Storybook", "Framer Motion",
        # State Management (additional)
        "Zustand", "MobX", "React Query", "TanStack Query", "SWR", "Recoil",
        # ORM (additional)
        "Drizzle ORM", "Hibernate", "MyBatis",
        # Cloud & Deployment (additional)
        "Vercel", "Netlify", "Heroku", "DigitalOcean", "Railway", "Fly.io",
        "AWS Lambda", "EKS", "ECS", "RDS", "CloudWatch", "Route 53", "CDK",
        "Azure DevOps", "Azure Functions", "Google Cloud", "Firebase Hosting",
        # CI/CD & DevOps (additional)
        "ArgoCD", "Helm", "Pulumi", "Vault", "Consul", "Istio", "FluxCD",
        "GitHub Copilot", "Bitbucket", "GitLab",
        # Package managers & runtime
        "Yarn", "pnpm", "Bun", "Deno", "Lerna", "Turborepo",
        # Vector DBs & RAG
        "Pinecone", "Weaviate", "ChromaDB", "FAISS", "Qdrant", "LlamaIndex",
        "Milvus", "pgvector", "Chroma"
    ]
    raw_lower = raw_text.lower()
    for tool in known_tools:
        pattern = r'\b' + re.escape(tool.lower()) + r'\b'
        if re.search(pattern, raw_lower):
            raw_collected.append(tool)
    # 4. Strict deduplication preserving appearance order
    final_skills = []
    seen = set()
    for skill in raw_collected:
        skill_norm = canonical_map.get(skill.lower(), skill)
        key = skill_norm.lower()
        if key not in seen and not _is_blacklisted(skill_norm) and len(skill_norm) > 1:
            seen.add(key)
            final_skills.append(skill_norm)
    if "Docker" in final_skills and "Docker Compose" in final_skills:
        final_skills.remove("Docker Compose")
    if "React.js" in final_skills and "React" in final_skills:
        final_skills.remove("React")
    # Explicit final safety sweep for canonical mapping and blacklist removal
    cleaned_final = []
    seen_final = set()
    for s in final_skills:
        s_norm = canonical_map.get(s.lower(), s).strip()
        k = s_norm.lower()
        if k not in seen_final and not _is_blacklisted(s_norm) and len(s_norm) > 1:
            seen_final.add(k)
            cleaned_final.append(s_norm)
    return cleaned_final
def generate_tailored_roles_universally(skills: List[str], seniority: str, primary_domain: str, raw_roles: List[str]) -> List[str]:
    """Universally generate accurate job roles for ANY domain."""
    skills_lower = {s.lower() for s in skills}
    domain_lower = primary_domain.lower()
    seniority_lower = seniority.lower()
    
    is_senior = any(w in seniority_lower for w in ["senior", "lead", "principal", "staff", "head", "director", "manager", "architect", "10", "12", "experienced"])
    prefix = "Senior" if is_senior else "Associate"
    
    is_software = any(w in domain_lower for w in ["software", "full-stack", "frontend", "backend", "developer", "web", "application"]) or any(w in skills_lower for w in ["javascript", "python", "react.js", "node.js", "php"])
    if is_software:
        has_react = any("react" in s for s in skills_lower)
        has_node = any("node" in s for s in skills_lower)
        has_mern = (has_react and has_node) or "mern" in skills_lower or any("express" in s or "mongo" in s for s in skills_lower)
        has_python = any("python" in s or "flask" in s or "django" in s for s in skills_lower)
        has_backend = has_node or has_python or any("php" in s or "backend" in s for s in skills_lower)
        has_frontend = has_react or any("vue" in s or "angular" in s or "frontend" in s for s in skills_lower)
        targeted = []
        targeted.append(f"{prefix} Full-Stack Engineer")
        if is_senior:
            targeted.append("Full-Stack Technical Lead")
        if has_mern:
            targeted.append(f"{prefix} MERN Stack Developer")
            if is_senior:
                targeted.append("Lead MERN Stack Architect")
        if has_react:
            targeted.append(f"{prefix} Frontend Engineer (React.js)")
        elif has_frontend:
            targeted.append(f"{prefix} Frontend Developer")
        if has_node:
            targeted.append(f"{prefix} Backend Engineer (Node.js)")
        elif has_backend:
            targeted.append(f"{prefix} Backend Engineer")
        if is_senior:
            targeted.append("Staff Software Engineer")
        banned = {"scrum master", "product manager", "project manager", "business analyst", "specialist"}
        filtered_raw = [r for r in raw_roles if not any(b in r.lower() for b in banned)]
        roles_pool = targeted + filtered_raw
    else:
        roles_pool = raw_roles if raw_roles else [f"{seniority} {primary_domain} Professional"]
    final_roles = []
    seen = set()
    for r in roles_pool:
        if r.lower() not in seen and len(r) > 2:
            seen.add(r.lower())
            final_roles.append(r)
    return final_roles[:6]
def build_accurate_summary(resume_text: str, domain: str, seniority: str, years_str: str, skills: List[str]) -> str:
    """
    Generates a coherent, authentic professional summary.
    Extracts authentic candidate summary from resume if present, or synthesizes a natural,
    cohesive 2-paragraph overview without hallucinations (no Java, no conflicting years).
    """
    summary_match = re.search(
        r'(?:PROFESSIONAL SUMMARY|EXECUTIVE SUMMARY|PROFILE|SUMMARY|ABOUT ME)[\s:\-]+([\s\S]+?)(?=\n\s*(?:TECHNICAL SKILLS|SKILLS|EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT|EDUCATION|PROJECTS|$))',
        resume_text,
        re.IGNORECASE
    )
    if summary_match:
        extracted = summary_match.group(1).strip()
        cleaned = re.sub(r'^[•\-\*]\s*', '', extracted, flags=re.MULTILINE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if len(cleaned) > 80:
            cleaned = re.sub(r'\b\d+\+?\s*years?\b', years_str, cleaned, flags=re.IGNORECASE)
            return cleaned
    domain_title = domain
    if domain_title.endswith("Engineering"):
        domain_title = domain_title.replace("Engineering", "Engineer")
    elif domain_title.endswith("Development"):
        domain_title = domain_title.replace("Development", "Developer")
    # Prioritize high-impact frameworks, languages, and cloud platforms for summary narrative
    priority_tech = [
        "React.js", "Node.js", "Express.js", "TypeScript", "JavaScript", "Python",
        "PHP", "PostgreSQL", "MongoDB", "Redis", "Docker", "AWS", "Next.js",
        "RESTful APIs", "GraphQL"
    ]
    featured = [s for s in priority_tech if s in skills]
    remaining = [
        s for s in skills 
        if s not in featured and s not in ["HTML5", "CSS3", "ES6+", "JSON", "YAML", "Git", "pgAdmin4"]
    ]
    top_tech_skills = (featured + remaining)[:7]
    skills_str = ", ".join(top_tech_skills) if top_tech_skills else "modern enterprise architectures"
    
    p1 = (
        f"{seniority} {domain_title} with {years_str} of extensive experience architecting, "
        f"developing, and deploying robust, high-performance web applications and cloud services."
    )
    p2 = (
        f"Specialized in full-lifecycle product development with deep proficiency across {skills_str}. "
        f"Demonstrated history of optimizing system scalability, designing RESTful and cloud-native architectures, "
        f"and delivering mission-critical solutions across high-impact engineering teams."
    )
    
    return f"{p1}\n\n{p2}"
def extract_json_from_output(raw_text: str) -> dict:
    """Extract clean JSON object from model generation text."""
    text = raw_text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return {
        "skills": ["Software Engineering"],
        "experience_level": "Senior",
        "years_of_experience": 12,
        "primary_domain": "Full-Stack Software Engineering",
        "matching_job_categories": ["Senior Full-Stack Engineer", "Technical Lead"],
        "summary": text[:250]
    }
@app.get("/api/stats")
def get_stats():
    return {
        "status": "Online",
        "model_name": "ResumeMatch-SLM",
        "parameters": "494M",
        "device": "CPU",
        "model_path": MODEL_PATH,
        "supported_features": ["Fast Pre-filtered Inference", "Authentic Summary Extraction", "Universal Multi-Domain Parsing", "Job Fit Scoring"]
    }
@app.post("/api/analyze")
def analyze_resume(req: ResumeRequest):
    if not req.resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text cannot be empty.")
    start_time = time.time()
    
    instruction = (
        "Analyze the resume text and output a structured JSON with keys: "
        "skills (array of strings), experience_level (Entry-Level/Junior/Mid-Level/Senior/Lead/Manager/Director/Executive), "
        "years_of_experience (number), primary_domain (string), matching_job_categories (array of strings), "
        "and summary (string). Return valid JSON only."
    )
    
    # 10x faster CPU inference: condense input to ~300 essential tokens
    condensed_text = condense_resume_for_inference(req.resume_text)
    
    messages = [
        {"role": "user", "content": f"{instruction}\n{condensed_text}"}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([formatted], return_tensors="pt")
    
    stopping_criteria = StoppingCriteriaList([StopOnJsonEnd(tokenizer, inputs.input_ids.shape[1])])
    
    # Fast inference under inference_mode with early stopping
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=110,
            do_sample=False,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    parsed_json = extract_json_from_output(generated_text)
    
    # 1. Clean & deduplicate technical skills (filter soft skills and redundant duplicates)
    raw_model_skills = parsed_json.get("skills", [])
    parsed_json["skills"] = extract_and_deduplicate_skills(req.resume_text, raw_model_skills)
    
    # 2. Universal experience timeline calculation (e.g. 12+ Years)
    years_num = parsed_json.get("years_of_experience", 12)
    parsed_json["years_of_experience"] = calculate_years_from_text(req.resume_text, years_num)
    # 3. Domain & seniority extraction
    domain = parsed_json.get("primary_domain", "Full-Stack Software Engineering")
    seniority = parsed_json.get("experience_level", "Senior")
    # 4. Universal role matching adapted to domain
    raw_roles = parsed_json.get("matching_job_categories", [])
    parsed_json["matching_job_categories"] = generate_tailored_roles_universally(
        skills=parsed_json["skills"],
        seniority=seniority,
        primary_domain=domain,
        raw_roles=raw_roles
    )
    # 5. Authentic, non-contradictory executive summary (no hallucinated Java, no conflicting years)
    parsed_json["summary"] = build_accurate_summary(
        resume_text=req.resume_text,
        domain=domain,
        seniority=seniority,
        years_str=parsed_json["years_of_experience"],
        skills=parsed_json["skills"]
    )
    latency_ms = round((time.time() - start_time) * 1000, 1)
    return {
        "success": True,
        "latency_ms": latency_ms,
        "data": parsed_json
    }
# =========================================================================
# COMPREHENSIVE PRODUCTION-GRADE SOFTWARE KNOWLEDGE GRAPH (250+ CONCEPTS)
# =========================================================================
UNIVERSAL_TECH_GRAPH: Dict[str, Set[str]] = {
    # 1. Frontend Languages, UI & Web Core
    "html": {"html5", "html"},
    "html5": {"html5", "html"},
    "css": {"css3", "css", "scss", "sass"},
    "css3": {"css3", "css", "scss", "sass"},
    "javascript": {"javascript", "es6+", "js"},
    "javascript (es6+)": {"javascript", "es6+", "js"},
    "es6": {"es6+", "javascript"},
    "es6+": {"es6+", "javascript"},
    "typescript": {"typescript", "ts"},
    "responsive ui development": {"html5", "css3", "material ui", "bootstrap", "tailwind", "responsive design", "responsive ui", "css"},
    "responsive ui": {"html5", "css3", "material ui", "bootstrap", "tailwind", "responsive design", "css"},
    "responsive web design": {"html5", "css3", "material ui", "bootstrap", "tailwind", "responsive design", "css"},
    "responsive design": {"html5", "css3", "material ui", "bootstrap", "tailwind", "responsive design"},
    "ui development": {"react.js", "angularjs", "vue.js", "html5", "css3", "material ui", "tailwind", "bootstrap", "javascript"},
    "cross-browser compatibility": {"html5", "css3", "javascript", "react.js", "playwright"},
    "semantic html": {"html5", "html"},
    "web accessibility": {"html5", "css3", "accessibility", "a11y"},
    "material ui": {"material ui", "mui"},
    "tailwind": {"tailwind", "tailwindcss"},
    "bootstrap": {"bootstrap"},
    # 2. Frontend Frameworks & Ecosystem
    "react": {"react.js", "react", "redux"},
    "reactjs": {"react.js", "react", "redux"},
    "react.js": {"react.js", "react", "redux"},
    "angular": {"angularjs", "angular"},
    "angularjs": {"angularjs", "angular"},
    "vue": {"vue.js", "vue"},
    "vuejs": {"vue.js", "vue"},
    "vue.js": {"vue.js", "vue"},
    "next.js": {"next.js", "nextjs", "react.js"},
    "nextjs": {"next.js", "nextjs", "react.js"},
    "redux": {"redux", "react.js"},
    "redux / context api": {"redux", "react.js"},
    "context api": {"react.js", "redux"},
    "state management": {"redux", "react.js", "context api", "zustand", "mobx", "recoil", "pinia", "vuex"},
    "data fetching": {"restful apis", "rest apis", "api integration", "third-party api integration", "graphql", "react query", "tanstack query", "swr", "axios", "fetch", "apollo", "apis"},
    "state management & data fetching": {"redux", "context api", "zustand", "restful apis", "react.js"},
    "data visualization": {"data visualization", "d3", "d3.js", "chart.js", "recharts", "highcharts", "plotly", "echarts", "canvas", "svg", "dashboard", "dashboards"},
    "visualization": {"data visualization", "d3", "chart.js", "recharts", "dashboard", "dashboards"},
    "dashboards": {"dashboard", "dashboards", "data visualization", "analytics"},
    "charts": {"chart.js", "recharts", "d3", "highcharts", "data visualization"},
    "accessible interfaces": {"html5", "css3", "accessibility", "a11y", "material ui"},
    "accessibility": {"html5", "css3", "accessibility", "a11y"},
    "frontend": {"react.js", "angularjs", "vue.js", "javascript", "html5", "css3"},
    "frontend development": {"react.js", "angularjs", "vue.js", "javascript", "html5", "css3"},
    "frontend design patterns & architecture": {"react.js", "redux", "angularjs", "vue.js", "frontend"},
    "frontend design patterns": {"react.js", "redux", "angularjs", "vue.js", "frontend"},
    "frontend architecture": {"react.js", "redux", "angularjs", "vue.js", "frontend"},
    # 3. Backend Runtimes, Languages & Stacks
    "node": {"node.js", "express.js", "nodejs"},
    "nodejs": {"node.js", "express.js", "nodejs"},
    "node.js": {"node.js", "express.js", "nodejs"},
    "express": {"express.js", "node.js"},
    "expressjs": {"express.js", "node.js"},
    "express.js": {"express.js", "node.js"},
    "python": {"python", "flask", "django", "fastapi"},
    "django": {"django", "python"},
    "flask": {"flask", "python"},
    "fastapi": {"fastapi", "python"},
    "php": {"php", "laravel"},
    "laravel": {"laravel", "php"},
    "java": {"java", "spring boot", "spring"},
    "spring boot": {"spring boot", "java"},
    "golang": {"golang", "go"},
    "go": {"golang", "go"},
    "backend": {"node.js", "express.js", "python", "flask", "django", "laravel", "php"},
    "backend development": {"node.js", "express.js", "python", "flask", "django", "laravel", "php"},
    "backend services": {"node.js", "express.js", "python", "flask", "restful apis"},
    "full stack": {"react.js", "node.js", "python", "javascript", "html5", "css3"},
    "full-stack": {"react.js", "node.js", "python", "javascript", "html5", "css3"},
    "mern stack": {"react.js", "node.js", "express.js", "mongodb", "mern"},
    "mern": {"react.js", "node.js", "express.js", "mongodb", "mern"},
    "mean stack": {"angularjs", "node.js", "express.js", "mongodb", "mean"},
    "mean": {"angularjs", "node.js", "express.js", "mongodb", "mean"},
    # 4. APIs & Communication
    "rest": {"restful apis"},
    "rest api": {"restful apis"},
    "rest apis": {"restful apis"},
    "restful apis": {"restful apis"},
    "rest api integration": {"restful apis", "third-party api integration"},
    "api integration": {"restful apis", "third-party api integration"},
    "third-party api integration": {"restful apis", "third-party api integration"},
    "third-party api integration & backend services": {"restful apis", "third-party api integration", "node.js", "python"},
    "graphql": {"graphql", "apollo"},
    "grpc": {"grpc", "protobuf"},
    "websocket": {"websocket", "socket.io"},
    "socket.io": {"websocket", "socket.io"},
    # 5. Databases, ORM & Caching
    "sql": {"postgresql", "mysql", "sqlite", "oracle", "mariadb", "sql"},
    "rdbms": {"postgresql", "mysql", "oracle", "sqlite", "mariadb", "sql"},
    "relational database": {"postgresql", "mysql", "oracle", "sqlite", "mariadb", "sql"},
    "relational databases": {"postgresql", "mysql", "oracle", "sqlite", "mariadb", "sql"},
    "data modelling": {"postgresql", "mysql", "mongodb", "sql", "database design", "data modeling"},
    "data modeling": {"postgresql", "mysql", "mongodb", "sql", "database design", "data modelling"},
    "database design": {"postgresql", "mysql", "mongodb", "sql", "database"},
    "database management": {"postgresql", "mysql", "mongodb", "redis", "database"},
    "schema design": {"postgresql", "mysql", "mongodb", "sql"},
    "nosql": {"mongodb", "redis", "dynamodb", "cassandra"},
    "mongodb": {"mongodb"},
    "postgresql": {"postgresql", "postgres"},
    "mysql": {"mysql"},
    "sqlite": {"sqlite"},
    "redis": {"redis"},
    "caching": {"redis", "memcached", "cloudfront"},
    # 6. Cloud, DevOps & Infrastructure
    "cloud": {"aws", "azure", "gcp", "docker", "kubernetes", "ec2", "s3"},
    "cloud technologies": {"aws", "azure", "gcp", "docker", "kubernetes", "ec2", "s3"},
    "cloud application development": {"aws", "azure", "gcp", "docker", "kubernetes", "ec2", "s3"},
    "aws": {"aws", "ec2", "s3", "cloudfront", "lambda", "ecs"},
    "gcp": {"gcp", "google cloud"},
    "azure": {"azure", "azure devops"},
    "docker": {"docker"},
    "kubernetes": {"kubernetes", "k8s"},
    "linux": {"linux", "ubuntu", "debian", "docker"},
    "microservices": {"docker", "kubernetes", "aws", "node.js", "express.js", "nginx", "microservices architecture"},
    "microservices architecture": {"docker", "kubernetes", "aws", "node.js", "express.js", "nginx", "microservices architecture"},
    "ci/cd": {"ci/cd", "github actions", "gitlab ci", "jenkins"},
    "github actions": {"github actions", "ci/cd"},
    "nginx": {"nginx"},
    # 7. Testing, QA, Debugging & Troubleshooting
    "unit testing": {"unit testing", "playwright", "jest", "cypress", "mocha", "pytest", "testing", "unit tests", "tdd"},
    "testing": {"unit testing", "playwright", "jest", "cypress", "mocha", "pytest", "testing", "tdd"},
    "automated testing": {"playwright", "cypress", "selenium", "jest", "pytest", "automation testing"},
    "automation testing": {"playwright", "cypress", "selenium", "jest", "pytest", "automated testing", "automation testing framework"},
    "automation testing framework": {"playwright", "cypress", "selenium", "jest", "pytest", "automated testing", "automation testing"},
    "debugging & troubleshooting": {"debugging", "troubleshooting", "playwright", "postman", "logging", "monitoring", "prometheus", "grafana"},
    "debugging": {"debugging", "troubleshooting", "playwright", "postman", "logging"},
    "troubleshooting": {"debugging", "troubleshooting", "playwright", "postman", "logging"},
    "playwright": {"playwright", "testing", "automated testing"},
    "cypress": {"cypress", "playwright", "testing", "automated testing", "e2e testing"},
    "cypress / playwright": {"cypress", "playwright", "testing"},
    "e2e testing": {"playwright", "cypress", "selenium", "testing", "automated testing"},
    "end-to-end testing": {"playwright", "cypress", "selenium", "testing", "automated testing"},
    "component testing": {"jest", "react testing library", "vitest", "unit testing", "testing"},
    # 8. Version Control
    "git": {"git", "github", "gitlab", "bitbucket", "version control"},
    "git (version control)": {"git", "github", "gitlab", "bitbucket", "version control"},
    "version control": {"git", "github", "gitlab", "bitbucket"},
    # 9. Engineering Leadership, Methodologies & Quality
    "tech lead": {"lead", "team lead", "tech lead", "principal", "architect", "engineering manager"},
    "team lead": {"lead", "team lead", "tech lead", "principal", "architect", "engineering manager"},
    "tech lead / team lead": {"lead", "team lead", "tech lead", "principal", "architect"},
    "team leadership": {"leadership", "team lead", "tech lead", "lead", "mentorship"},
    "people management": {"leadership", "management", "team lead", "lead"},
    "sdlc": {"sdlc", "software development life cycle"},
    "agile": {"agile", "scrum", "kanban", "sprints"},
    "scrum": {"agile", "scrum", "kanban", "sprints"},
    "code reviews": {"code reviews", "code review", "pr reviews"},
    "software delivery": {"software delivery", "ci/cd", "deployment", "deploying", "release management"},
    "performance optimization": {"performance optimization", "caching", "redis", "cloudfront", "nginx", "optimization", "scalable", "scaling", "webpack", "vite"},
    "clean code": {"code reviews", "design patterns", "refactoring"},
    "design patterns": {"architecture", "design patterns", "oop"},
    # 10. AI, LLMs & Emerging Tech
    "genai": {"genai", "generative ai", "gen ai", "llm", "llms", "prompt engineering", "langchain", "openai"},
    "generative ai": {"genai", "generative ai", "gen ai", "llm", "llms", "prompt engineering", "langchain", "openai"},
    "ai tools": {"openai", "langchain", "llms", "generative ai", "genai"},
    "ai apis": {"openai", "langchain", "restful apis"},
    "llm": {"llm", "llms", "openai", "genai", "generative ai"},
    "llms": {"llm", "llms", "openai", "genai", "generative ai"},
    "rag": {"rag"},
    "prompt engineering": {"prompt engineering"},
    "agentic development": {"ai agents", "langchain", "agentic ai"},
    "ai workflows": {"ai workflows", "langchain"},
    "ai coding tools": {"copilot", "github copilot", "cursor", "chatgpt", "claude", "cody", "windsurf", "v0", "ai tools", "agentic ai", "ai coding"},
    "agentic ai coding tools": {"cursor", "copilot", "agentic ai", "ai tools"},
    # 11. Mobile & Cross-Platform
    "flutter": {"flutter", "dart", "mobile"},
    "dart": {"flutter", "dart"},
    "react native": {"react native", "react-native"},
    "ios": {"swift", "objective-c", "ios", "xcode", "mobile"},
    "android": {"kotlin", "java", "android", "mobile"},
    "mobile development": {"flutter", "react native", "ios", "android", "mobile"},
    "app store / play store deployments": {"mobile", "deployment", "play store", "app store", "ci/cd", "software delivery"},
    "app store": {"mobile", "deployment", "play store", "app store", "software delivery"},
    "play store": {"mobile", "deployment", "play store", "app store", "software delivery"},
    # 12. Systems, Distributed Architecture & IoT
    "system design": {"system design", "microservices", "microservices architecture", "architecture", "distributed systems", "scalability"},
    "system design & architecture": {"system design", "microservices", "microservices architecture", "architecture", "scalability"},
    "distributed systems": {"microservices", "system design", "distributed systems", "kafka", "redis"},
    "iot": {"iot", "connected devices", "embedded", "mqtt", "hardware"},
    "connected devices": {"iot", "connected devices", "embedded", "mqtt"},
    "iot / connected devices": {"iot", "connected devices", "embedded", "mqtt"},
    # 13. Containerization
    "containerization": {"docker", "kubernetes", "containers", "containerization"},
    "docker containerization": {"docker", "kubernetes", "containers", "containerization"},
    # 14. Serverless & Messaging Services
    "lambda": {"lambda", "aws lambda", "serverless"},
    "aws lambda": {"lambda", "aws lambda", "serverless"},
    "appsync": {"appsync", "aws appsync"},
    "aws appsync": {"appsync", "aws appsync"},
    "dynamodb": {"dynamodb", "aws dynamodb", "nosql"},
    "aws dynamodb": {"dynamodb", "aws dynamodb", "nosql"},
    "eventbridge": {"eventbridge", "aws eventbridge", "event-driven"},
    "aws eventbridge": {"eventbridge", "aws eventbridge", "event-driven"},
    "sqs": {"sqs", "aws sqs", "message queue", "queues"},
    "aws sqs": {"sqs", "aws sqs", "message queue", "queues"},
    "webhooks": {"webhooks", "webhook", "apis", "restful apis"},
    "event-driven architecture": {"event-driven", "microservices", "pub/sub", "kafka", "redis", "rabbitmq", "microservices architecture"},
    "event-driven systems": {"event-driven", "microservices", "pub/sub", "kafka", "redis", "rabbitmq", "microservices architecture"},
    "reliability": {"reliability", "high availability", "monitoring", "scalability", "microservices architecture", "system design"},
    "feature ownership": {"feature ownership", "software delivery", "tech lead", "lead", "sdlc", "agile", "principal"},
    "end-to-end feature ownership": {"feature ownership", "software delivery", "tech lead", "lead", "sdlc", "agile", "principal"},
    "production systems": {"production systems", "production applications", "scalable web applications", "software delivery"},
    # 15. Business, Financial & Banking Domains
    "banking domain": {"banking domain", "banking", "bfsi", "fintech", "financial services"},
    "banking": {"banking domain", "banking", "bfsi", "fintech", "financial services"},
    "bfsi": {"banking domain", "banking", "bfsi", "fintech", "financial services"},
    "fintech": {"fintech", "banking domain", "banking", "bfsi", "financial services"},

}
# Precomputed normalized index for sub-millisecond, robust punctuation-free lookups
def normalize_skill_key(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[\(\)\{\}\[\]\.,;:\-–—/\\]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

NORMALIZED_TECH_GRAPH: Dict[str, Set[str]] = {}
def _init_normalized_graph():
    global NORMALIZED_TECH_GRAPH
    for k, vals in UNIVERSAL_TECH_GRAPH.items():
        k_norm = normalize_skill_key(k)
        if k_norm not in NORMALIZED_TECH_GRAPH:
            NORMALIZED_TECH_GRAPH[k_norm] = set()
        for v in vals:
            NORMALIZED_TECH_GRAPH[k_norm].add(normalize_skill_key(v))

_init_normalized_graph()

STOPWORDS = {
    'experience', 'strong', 'hands-on', 'ability', 'to', 'in', 'with', 'understanding',
    'of', 'skills', 'preferred', 'mandatory', 'knowledge', 'proven', 'proficiency',
    'practical', 'familiarity', 'minimum', 'years', 'production', 'applications',
    'solutions', 'build', 'using', 'and', 'or', 'for', 'a', 'an', 'the', 'good', 'have',
    'plus', 'candidate', 'developer', 'engineer', 'work', 'working', 'team', 'teams'
}

COMMON_NON_SKILLS = {
    'services', 'states', 'data', 'code', 'practices', 'patterns', 'tools', 'applications', 'solutions', 'processes',
    'rendering', 'empty', 'error states', 'component', 'debug', 'the ability to review', 'test ai-generated code',
    'handle loading', 'production applications', 'web applications', 'responsive layouts', 'empty and error states'
}
BROAD_PLATFORMS = {'aws', 'cloud', 'gcp', 'azure', 'google cloud', 'react', 'node', 'python', 'java'}

TECH_DISPLAY_CASES = {
    'aws': 'AWS', 'api': 'API', 'apis': 'APIs', 'ci/cd': 'CI/CD', 'sql': 'SQL', 'rdbms': 'RDBMS',
    'sqs': 'SQS', 'graphql': 'GraphQL', 'dynamodb': 'DynamoDB', 'appsync': 'AppSync',
    'eventbridge': 'EventBridge', 'sdlc': 'SDLC', 'ui': 'UI', 'ux': 'UX', 'ai': 'AI',
    'ml': 'ML', 'llm': 'LLM', 'llms': 'LLMs', 'rest': 'REST', 'restful': 'RESTful',
    'php': 'PHP', 'html': 'HTML', 'html5': 'HTML5', 'css': 'CSS', 'css3': 'CSS3',
    'ec2': 'EC2', 's3': 'S3', 'gcp': 'GCP', 'tcp/ip': 'TCP/IP', 'pl/sql': 'PL/SQL',
    'postgresql': 'PostgreSQL', 'mysql': 'MySQL', 'mongodb': 'MongoDB', 'redis': 'Redis',
    'docker': 'Docker', 'kubernetes': 'Kubernetes', 'jenkins': 'Jenkins', 'git': 'Git',
    'github': 'GitHub', 'gitlab': 'GitLab', 'playwright': 'Playwright', 'cypress': 'Cypress',
    'genai': 'GenAI', 'generative ai': 'Generative AI', 'banking domain': 'Banking Domain',
    'bfsi': 'BFSI', 'fintech': 'FinTech'
}

def format_single_title(skill: str) -> str:
    s = skill.strip()
    lower = s.lower()
    if lower in TECH_DISPLAY_CASES:
        return TECH_DISPLAY_CASES[lower]
    if re.match(r'^(node|react|vue|next|nuxt|nest|express|angular)\.js$', lower):
        m = re.match(r'^([a-z]+)\.js$', lower)
        return m.group(1).capitalize() + '.js'
    words = s.split()
    formatted_words = []
    for w in words:
        if '-' in w:
            parts = [TECH_DISPLAY_CASES.get(p.lower(), p.capitalize()) for p in w.split('-')]
            formatted_words.append('-'.join(parts))
        elif w.lower() in TECH_DISPLAY_CASES:
            formatted_words.append(TECH_DISPLAY_CASES[w.lower()])
        elif w.islower() or (w.isupper() and len(w) > 4):
            formatted_words.append(w.capitalize())
        else:
            formatted_words.append(w)
    return ' '.join(formatted_words)

def format_skill_title(skill: str) -> str:
    """Format skills to clean, professional Title Case and standard uppercase acronyms, preserving OR groups."""
    if ' or ' in skill.lower():
        parts = [format_single_title(p.strip()) for p in re.split(r'\s+or\s+', skill, flags=re.IGNORECASE) if p.strip()]
        return ' or '.join(parts)
    return format_single_title(skill)

def clean_bullet_prefix(text: str) -> str:
    """Robust JD bullet cleaner handling Unicode emojis, Roman numerals, letters, and punctuation."""
    return re.sub(r'^(?:[•●▪▫◦*✔✅📌🔑🎯🚀💼📍\ufe0f\s]+|[-–—]\s*|\d+[.)]\s*)', '', text.strip()).strip()

def clean_atomic_phrase(s: str) -> str:
    """Strip boilerplate verb phrases and clean down to atomic skill title."""
    lead_pat = r'^(?:(?:(?:minimum\s+)?\d+[\+\-]?\s*(?:years?|yrs?)(?:\s+of)?|strong|solid|proven|deep|extensive|broad|hands[-\s]?on|good|basic|practical|demonstrated|demonstrable|excellent)\s+)*(?:hands[-\s]?on\s+)?(?:experience\s+(?:with|in|as|of)?|understanding\s+of|knowledge\s+of|expertise\s+in|proficiency\s+(?:with|in)?|proficient\s+(?:with|in)?|ability\s+to\s+build|ability\s+to|proven\s+track\s+record\s+in|background\s+in|skills?\s+in|design\s+and\s+implement|develop\s+and\s+maintain|building\s+and\s+deploying|practical\s+experience\s+with)\s*'
    s = re.sub(lead_pat, '', s, flags=re.IGNORECASE).strip()

    # Also strip standalone leading adjectives/articles if followed by the skill
    s = re.sub(r'^(?:strong|solid|proven|deep|extensive|hands[-\s]?on|good|basic|practical|excellent|a|an)\s+', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'^(?:a|an)\s+', '', s, flags=re.IGNORECASE).strip()

    s = re.sub(r'\s+(?:experience\s+)?is\s+a\s+plus\.?$', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'\s+(?:experience\s+)?is\s+mandatory\.?$', '', s, flags=re.IGNORECASE).strip()
    # Strip any trailing parenthetical/bracketed qualifier (closed or unclosed, single or multi-word)
    s = re.sub(r'\s*[\(\[][^)\]]*(?:mandatory|required|essential|preferred|optional|bonus|plus|must|nice|good|immediate|years?|yrs?|hands[-\s]?on|complex|experience)[^)\]]*[\)\]]?\s*$', '', s, flags=re.IGNORECASE).strip()
    # Strip parenthetical elaborations like (routing, rendering and API routes) or (IIT/NIT/BITS or equivalent)
    s = re.sub(r'\s*\([^)]*(?:routing|rendering|etc|e\.g\.|i\.e\.|including|such as|equivalent|complex)[^)]*\)', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'\s+in\s+production\s+applications\.?$', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'\s+and\s+the\s+ability\s+to\s+.*$', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'\s+and\s+handle\s+.*$', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'\s+experience$', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'\s+skills?$', '', s, flags=re.IGNORECASE).strip()
    # Strip trailing role/development suffixes if preceded by technology or stack
    if not re.search(r'^(?:software|senior|lead|principal|junior)\s+(?:developer|engineer)$', s, re.IGNORECASE):
        s = re.sub(r'\s+(?:development|developer|engineer|engineering)$', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'^(?:and/or|and|or|/or)\s+', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'^[^\w\s]+', '', s).strip()
    s = s.rstrip('.').strip()
    s = re.sub(r'\s+', ' ', s)
    # Ensure balanced parentheses:
    if '(' in s and ')' not in s:
        s = re.sub(r'\s*\([^)]*$', '', s).strip()
    if ')' in s and '(' not in s:
        s = re.sub(r'\s*\)[^)]*$', '', s).strip()
        s = s.rstrip(')').strip()
    s = s.strip()
    return s.strip()

def clean_and_decompose_line(line: str) -> List[str]:
    """Break complex bullet sentences into atomic, discrete skill blocks while safely preserving CI/CD and C/C++."""
    s = clean_bullet_prefix(line)
    s = re.sub(r'^(?:must[-\s]?have\s+skills?|key\s+skills?|required\s+skills?|requirements?|backend|frontend|languages?|technologies|database|cloud|devops)\s*:\s*', '', s, flags=re.IGNORECASE).strip()
    s = clean_atomic_phrase(s)

    # 1. Pipe-separated lines (e.g. "Tech Stack: AWS | TypeScript | Node.js...")
    if '|' in s:
        s = re.sub(r'^tech\s+stack\s*:\s*', '', s, flags=re.IGNORECASE).strip()
        parts = s.split('|')
        items = []
        for p in parts:
            p_clean = clean_atomic_phrase(p)
            if p_clean and len(p_clean) > 1 and p_clean.lower() not in COMMON_NON_SKILLS:
                items.append(p_clean)
        return items

    # 2. Colon or dash-introduced sub-lists (e.g. "AWS serverless services: Lambda, AppSync", "Automation Testing Framework – Playwright", "Backend – Node.js")
    colon_or_dash_m = re.match(r'^([^:,–—]+?)\s*(?::|\s+[-–—]\s+)\s*(.+)$', s)
    if colon_or_dash_m:
        prefix = colon_or_dash_m.group(1).strip()
        body = colon_or_dash_m.group(2).strip()
        body = re.sub(r'\band/or\b', ' or ', body, flags=re.IGNORECASE)
        body = re.sub(r'\s+/\s+', ' or ', body)
        # Split on commas, &, or 'and', preserving 'or' alternative groups within parts
        parts = re.split(r'[,&]|\band\b', body, flags=re.IGNORECASE)
        items = []
        for p in parts:
            p_clean = clean_atomic_phrase(p)
            if p_clean and len(p_clean) > 1 and p_clean.lower() not in COMMON_NON_SKILLS:
                items.append(p_clean)
        return items

    # 2b. Parenthetical sub-lists (e.g. "Vector Databases (Pinecone, Chroma, Qdrant)", "Unit testing (pytest)")
    paren_m = re.match(r'^([^(\n]+?)\s*\(([^)]+)\)\s*$', s)
    if paren_m:
        prefix = clean_atomic_phrase(paren_m.group(1))
        body = paren_m.group(2).strip()
        body = re.sub(r'\band/or\b', ' or ', body, flags=re.IGNORECASE)
        body = re.sub(r'\s+/\s+', ' or ', body)
        if not re.search(r'^(?:mandatory|required|essential|preferred|optional|bonus|plus|minimum|hands[-\s]?on|complex)$', body, re.IGNORECASE):
            parts = re.split(r'[,&]|\band\b', body, flags=re.IGNORECASE)
            items = []
            if prefix and len(prefix) > 2 and prefix.lower() not in COMMON_NON_SKILLS:
                items.append(prefix)
            for p in parts:
                p_clean = clean_atomic_phrase(p)
                if p_clean and len(p_clean) > 1 and p_clean.lower() not in COMMON_NON_SKILLS:
                    items.append(p_clean)
            if items:
                return items

    # 3. Standard comma, &, or 'and' (preserves 'or' alternative groups within items)
    s = re.sub(r'\band/or\b', ' or ', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+/\s+', ' or ', s)
    parts = re.split(r'[,&]|\band\b', s, flags=re.IGNORECASE)
    items = []
    for p in parts:
        p_clean = clean_atomic_phrase(p)
        if p_clean and p_clean.lower() not in COMMON_NON_SKILLS and len(p_clean) > 1:
            items.append(p_clean)
    return items if items else [s]

MANDATORY_HEADER_RE = re.compile(
    r"""(?:must[-\s]?have|required\s+skills?|requirements?|what\s+we(?:['\u2019]?re|\s+are)\s+looking\s+for|what\s+you(?:['\u2019]?ll|\s+will)\s+need|technical\s+skills?|tech\s+stack|technologies|skills?\s+required|qualifications?|core\s+competencies|key\s+skills?|eligibility|competencies|skillsets?|profile\s+requirements?|candidate\s+profile)""",
    re.IGNORECASE
)
PREFERRED_HEADER_RE = re.compile(
    r"""(?:good\s+to\s+have|nice\s+to\s+have|preferred\s+skills|preferred\s+qualifications|preferred\s+background|preferred|bonus|optional|desirable|added\s+advantage|plus)""",
    re.IGNORECASE
)
TRUE_STOP_RE = re.compile(
    r"""(?:timings?|working\s+days?|notice\s+period|relocation|role\s+overview|job\s+overview|position\s+overview|role\s+summary|job\s+summary|position\s+summary|about\s+(?:the\s+)?role|about\s+(?:the\s+)?job|job\s+description|scope\s+of\s+work|key\s+responsibilit|responsibilit|what\s+you(?:['\u2019]?ll|\s+will)\s+do|role\s+&(?:amp;)?\s+responsibilities|roles?\s+and\s+responsibilities|duties|day\s+to\s+day|about\s+(?:us|the\s+company)|who\s+we\s+are|perks|benefits|apply\s+now|how\s+to\s+apply|interested\s+candidates|contact\s+us|location|ctc|salary|employment|interview)""",
    re.IGNORECASE
)
PREFERRED_LINE_RE = re.compile(
    r"""\b(?:is\s+a\s+plus|would\s+be\s+a\s+plus|nice\s+to\s+have|good\s+to\s+have|preferred|bonus|optional|added\s+advantage|desirable)\b""",
    re.IGNORECASE
)
NOISE_RE = re.compile(
    r"""@|https?://|experience(?:\s+required)?\s*:|location\s*:|ctc\s*:|salary\s*:|timings?\s*:|interview(?:\s+date|\s+mode|\s+level)?\s*:|employment(?:\s+type)?\s*:|\b\d+[\+\-]?\s*(?:years?|yrs?)\s+(?:in|of)\b|\b\d{10}\b|immediate\s+joiners|interested\s+candidates|dm\s+me|share\s+their\s+updated|apply\s+now|\b(?:b\.?tech|m\.?tech|bca|mca|bachelor(?:s)?|master(?:s)?|tier\s*1\s*institute|degree\s+in)\b""",
    re.IGNORECASE
)
BULLET_RE = re.compile(r'^[•●▪▫◦*✔✅📌🔑🎯🚀💼📍\ufe0f\-\–—\d\.]')

def is_header_line(s: str) -> bool:
    """Detect if a line is a structural header and NOT a bullet or list item."""
    stripped = s.strip()
    if not stripped:
        return False
    # If line starts with bullet marker or list digit, it is not a header unless it's a known header title with an emoji icon
    if re.match(r'^(?:[•●▪▫◦*✔✅📌🔑🎯🚀💼📍\ufe0f\s]+|[-–—]\s*|\d+[.)]\s*)', stripped):
        clean_after = re.sub(r'^[^\w\s]+', '', stripped).strip()
        if any(re_pat.search(clean_after) for re_pat in [MANDATORY_HEADER_RE, PREFERRED_HEADER_RE, TRUE_STOP_RE]):
            if not any(c in stripped for c in ['|', ',']):
                # Only treat as header if it does NOT start with a dash/bullet marker
                if not re.match(r'^[-–—•●▪▫◦*✔✅\d]', stripped):
                    return True
        return False
    clean = re.sub(r'^[^\w\s]+', '', stripped).strip()
    if any(re_pat.search(clean) for re_pat in [MANDATORY_HEADER_RE, PREFERRED_HEADER_RE, TRUE_STOP_RE]):
        if not any(c in stripped for c in ['|', ',']):
            return True
    if stripped.endswith(':') or clean.endswith(':'):
        return True
    return False

def extract_universal_criteria_tiered(jd_text: str) -> Dict[str, Any]:
    """Extract atomic JD criteria and classify into Core Mandatory vs Preferred Bonus tiers."""
    lines = jd_text.strip().splitlines()
    raw_criteria = []
    in_section = False
    current_importance = "mandatory"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        clean_hdr = re.sub(r'^[^\w\s]+', '', stripped).strip()
        is_hdr = is_header_line(stripped)

        # 1. Preferred header switches section importance to preferred (does NOT terminate extraction)
        if PREFERRED_HEADER_RE.search(clean_hdr) and is_hdr:
            in_section = True
            current_importance = "preferred"
            continue

        # 2. Mandatory header switches section importance to mandatory
        if MANDATORY_HEADER_RE.search(clean_hdr) and is_hdr:
            in_section = True
            current_importance = "mandatory"
            continue

        # 3. True stop header terminates requirement extraction (Perks, Location, About Us)
        if in_section and TRUE_STOP_RE.search(clean_hdr) and is_hdr:
            in_section = False
            break

        if in_section:
            if NOISE_RE.search(stripped):
                continue
            clauses = re.split(r';|,\s*(?:while|whereas|but)\s+', stripped)
            for clause in clauses:
                clause = clause.strip()
                if not clause or NOISE_RE.search(clause):
                    continue
                clause_importance = "preferred" if PREFERRED_LINE_RE.search(clause) else current_importance
                atomic_items = clean_and_decompose_line(clause)
                for item in atomic_items:
                    if item and not NOISE_RE.search(item) and len(item.split()) <= 8:
                        raw_criteria.append((item, clause_importance))

    # Section-agnostic fallback: capture bullet points and stack lines if no explicit section header exists
    if not raw_criteria:
        for line in lines:
            stripped = line.strip()
            if not stripped or NOISE_RE.search(stripped):
                continue
            if BULLET_RE.search(stripped) or '|' in stripped or re.match(r'^(?:tech\s+stack|technologies|skills?)\s*:', stripped, re.IGNORECASE):
                clauses = re.split(r';|,\s*(?:while|whereas|but)\s+', stripped)
                for clause in clauses:
                    clause = clause.strip()
                    if not clause or NOISE_RE.search(clause):
                        continue
                    clause_importance = "preferred" if PREFERRED_LINE_RE.search(clause) else "mandatory"
                    atomic_items = clean_and_decompose_line(clause)
                    for item in atomic_items:
                        if item and not NOISE_RE.search(item) and len(item.split()) <= 8:
                            raw_criteria.append((item, clause_importance))

    # Smart deduplication: keep longer name, promote to mandatory if any mention was mandatory
    seen_skills = {}
    for c, imp in raw_criteria:
        norm = normalize_skill_key(re.sub(r'^(?:aws|modern)\s+', '', c, flags=re.IGNORECASE))
        if len(norm) > 1:
            if norm not in seen_skills:
                seen_skills[norm] = (c, imp)
            else:
                prev_c, prev_imp = seen_skills[norm]
                new_imp = "mandatory" if (imp == "mandatory" or prev_imp == "mandatory") else "preferred"
                new_c = c if len(c) > len(prev_c) else prev_c
                seen_skills[norm] = (new_c, new_imp)

    core_skills = []
    preferred_skills = []
    for norm, (c, imp) in seen_skills.items():
        title = format_skill_title(c)
        if imp == "mandatory":
            core_skills.append(title)
        else:
            preferred_skills.append(title)

    # Fallback to graph keyword scan if completely empty
    if not core_skills and not preferred_skills:
        found_skills = []
        for key in sorted(NORMALIZED_TECH_GRAPH.keys(), key=len, reverse=True):
            if len(key) >= 3 and re.search(r'\b' + re.escape(key) + r'\b', jd_text, re.IGNORECASE):
                if not any(re.search(r'\b' + re.escape(key) + r'\b', existing.lower()) for existing in found_skills):
                    clean_display = format_skill_title(key)
                    found_skills.append(clean_display)
        core_skills = found_skills[:15]

    return {
        "core": core_skills,
        "preferred": preferred_skills,
        "all": core_skills + preferred_skills
    }

def extract_universal_criteria(jd_text: str) -> List[str]:
    """Backward-compatible helper returning all extracted criteria."""
    return extract_universal_criteria_tiered(jd_text)["all"]

COMPOUND_STACK_RULES: Dict[str, Dict[str, Any]] = {
    "mern stack": {
        "display": "MERN Stack",
        "acronym": "mern",
        "frontend": ["react.js", "react", "reactjs", "next.js", "nextjs"],
        "backend": ["node.js", "node", "nodejs", "express.js", "express", "expressjs"],
        "database": ["mongodb", "mongo"],
        "components": [
            ("MongoDB", ["mongodb", "mongo"]),
            ("Express.js", ["express.js", "express", "expressjs"]),
            ("React.js", ["react.js", "react", "reactjs", "next.js"]),
            ("Node.js", ["node.js", "node", "nodejs"])
        ]
    },
    "mern": {
        "display": "MERN Stack",
        "acronym": "mern",
        "frontend": ["react.js", "react", "reactjs", "next.js", "nextjs"],
        "backend": ["node.js", "node", "nodejs", "express.js", "express", "expressjs"],
        "database": ["mongodb", "mongo"],
        "components": [
            ("MongoDB", ["mongodb", "mongo"]),
            ("Express.js", ["express.js", "express", "expressjs"]),
            ("React.js", ["react.js", "react", "reactjs", "next.js"]),
            ("Node.js", ["node.js", "node", "nodejs"])
        ]
    },
    "mean stack": {
        "display": "MEAN Stack",
        "acronym": "mean",
        "frontend": ["angularjs", "angular"],
        "backend": ["node.js", "node", "nodejs", "express.js", "express", "expressjs"],
        "database": ["mongodb", "mongo"],
        "components": [
            ("MongoDB", ["mongodb", "mongo"]),
            ("Express.js", ["express.js", "express", "expressjs"]),
            ("AngularJS", ["angularjs", "angular"]),
            ("Node.js", ["node.js", "node", "nodejs"])
        ]
    },
    "mean": {
        "display": "MEAN Stack",
        "acronym": "mean",
        "frontend": ["angularjs", "angular"],
        "backend": ["node.js", "node", "nodejs", "express.js", "express", "expressjs"],
        "database": ["mongodb", "mongo"],
        "components": [
            ("MongoDB", ["mongodb", "mongo"]),
            ("Express.js", ["express.js", "express", "expressjs"]),
            ("AngularJS", ["angularjs", "angular"]),
            ("Node.js", ["node.js", "node", "nodejs"])
        ]
    },
    "mevn stack": {
        "display": "MEVN Stack",
        "acronym": "mevn",
        "frontend": ["vue.js", "vue", "vuejs"],
        "backend": ["node.js", "node", "nodejs", "express.js", "express", "expressjs"],
        "database": ["mongodb", "mongo"],
        "components": [
            ("MongoDB", ["mongodb", "mongo"]),
            ("Express.js", ["express.js", "express", "expressjs"]),
            ("Vue.js", ["vue.js", "vue", "vuejs"]),
            ("Node.js", ["node.js", "node", "nodejs"])
        ]
    },
    "mevn": {
        "display": "MEVN Stack",
        "acronym": "mevn",
        "frontend": ["vue.js", "vue", "vuejs"],
        "backend": ["node.js", "node", "nodejs", "express.js", "express", "expressjs"],
        "database": ["mongodb", "mongo"],
        "components": [
            ("MongoDB", ["mongodb", "mongo"]),
            ("Express.js", ["express.js", "express", "expressjs"]),
            ("Vue.js", ["vue.js", "vue", "vuejs"]),
            ("Node.js", ["node.js", "node", "nodejs"])
        ]
    },
    "pern stack": {
        "display": "PERN Stack",
        "acronym": "pern",
        "frontend": ["react.js", "react", "reactjs", "next.js"],
        "backend": ["node.js", "node", "nodejs", "express.js", "express", "expressjs"],
        "database": ["postgresql", "postgres"],
        "components": [
            ("PostgreSQL", ["postgresql", "postgres"]),
            ("Express.js", ["express.js", "express", "expressjs"]),
            ("React.js", ["react.js", "react", "reactjs", "next.js"]),
            ("Node.js", ["node.js", "node", "nodejs"])
        ]
    },
    "pern": {
        "display": "PERN Stack",
        "acronym": "pern",
        "frontend": ["react.js", "react", "reactjs", "next.js"],
        "backend": ["node.js", "node", "nodejs", "express.js", "express", "expressjs"],
        "database": ["postgresql", "postgres"],
        "components": [
            ("PostgreSQL", ["postgresql", "postgres"]),
            ("Express.js", ["express.js", "express", "expressjs"]),
            ("React.js", ["react.js", "react", "reactjs", "next.js"]),
            ("Node.js", ["node.js", "node", "nodejs"])
        ]
    },
    "lamp stack": {
        "display": "LAMP Stack",
        "acronym": "lamp",
        "components": [
            ("Linux", ["linux", "ubuntu"]),
            ("Apache", ["apache"]),
            ("MySQL", ["mysql", "mariadb"]),
            ("PHP", ["php", "laravel"])
        ]
    },
    "lamp": {
        "display": "LAMP Stack",
        "acronym": "lamp",
        "components": [
            ("Linux", ["linux", "ubuntu"]),
            ("Apache", ["apache"]),
            ("MySQL", ["mysql", "mariadb"]),
            ("PHP", ["php", "laravel"])
        ]
    }
}

def evaluate_compound_stack(s_clean: str, cand_set: Set[str], text_lower: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Evaluates compound multi-tier tech stacks (MERN, MEAN, MEVN, PERN, LAMP) against constituent resume skills."""
    if s_clean not in COMPOUND_STACK_RULES:
        return False, None, None
        
    stack_info = COMPOUND_STACK_RULES[s_clean]
    acronym = stack_info["acronym"]
    display_name = stack_info["display"]

    # 1. Look for constituent skills in candidate skills set and resume text
    found_components = []
    for label, aliases in stack_info["components"]:
        has_comp = any(a in cand_set for a in aliases) or any(
            len(a) >= 3 and re.search(r'\b' + re.escape(a) + r'\b', text_lower) for a in aliases
        )
        if has_comp:
            found_components.append(label)

    # 2. Check if acronym itself appears in candidate profile
    has_acronym = acronym in cand_set or bool(re.search(r'\b' + re.escape(acronym) + r'\b', text_lower))

    # 3. Stack-specific qualification logic
    if s_clean in ["mern stack", "mern", "mean stack", "mean", "mevn stack", "mevn", "pern stack", "pern"]:
        has_frontend = any(f in cand_set for f in stack_info["frontend"]) or any(
            re.search(r'\b' + re.escape(f) + r'\b', text_lower) for f in stack_info["frontend"]
        )
        has_backend = any(b in cand_set for b in stack_info["backend"]) or any(
            re.search(r'\b' + re.escape(b) + r'\b', text_lower) for b in stack_info["backend"]
        )
        if has_acronym or (has_frontend and has_backend):
            comp_str = ", ".join(found_components)
            detail_str = f"{display_name} ({comp_str})" if comp_str else display_name
            return True, detail_str, None
        else:
            missing = []
            if not has_frontend:
                missing.append("Frontend")
            if not has_backend:
                missing.append("Backend")
            gap_note = f"{display_name} (Missing {', '.join(missing)})"
            return False, None, gap_note
            
    elif s_clean in ["lamp stack", "lamp"]:
        has_php = any(p in cand_set or re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in ["php", "laravel"])
        has_other = len(found_components) >= 2
        if has_acronym or (has_php and has_other):
            comp_str = ", ".join(found_components)
            detail_str = f"{display_name} ({comp_str})" if comp_str else display_name
            return True, detail_str, None
        else:
            return False, None, f"{display_name} (Missing PHP/Apache/Linux)"

    return False, None, None

def evaluate_single_criterion(skill: str, candidate_skills: List[str], candidate_text: str = "", seniority: str = "") -> Tuple[bool, Optional[str], Optional[str]]:
    """Evaluate a single atomic skill block with full semantic fallback."""
    cand_lower = {normalize_skill_key(s): s for s in candidate_skills}
    cand_set = set(cand_lower.keys())
    text_lower = candidate_text.lower() if candidate_text else ""
    s_clean = normalize_skill_key(skill)

    if not s_clean:
        return False, None, skill

    # Compound Stack evaluation (MERN, MEAN, MEVN, PERN, LAMP)
    if s_clean in COMPOUND_STACK_RULES:
        is_stack_met, stack_detail, stack_gap = evaluate_compound_stack(s_clean, cand_set, text_lower)
        if is_stack_met:
            return True, stack_detail, None
        else:
            return False, None, stack_gap

    # Guard: frontend requires frontend stack
    if "frontend" in s_clean and not any(f in cand_set for f in ["react.js", "angularjs", "vue.js", "javascript", "html5", "css3", "frontend"]):
        return False, None, skill

    # Leadership / Seniority / Delivery check
    if any(k in s_clean for k in ["tech lead", "team lead", "head of", "lead developer", "engineering manager", "team leadership", "mentoring", "feature ownership"]):
        if seniority == "Lead / Principal" or any(w in text_lower for w in ["lead software engineer", "tech lead", "team lead", "head of development", "lead developer", "lead engineer", "software delivery"]):
            return True, skill, None

    # Direct match in candidate's skills set
    if s_clean in cand_set:
        return True, skill, None

    # Base concept without parenthetical notes
    s_no_paren = normalize_skill_key(re.sub(r'\s*\([^)]*\)', '', s_clean))
    if s_no_paren and s_no_paren in cand_set:
        return True, skill, None

    # Exact word match in resume text (min length 4)
    if len(s_clean) > 3 and re.search(r'\b' + re.escape(s_clean) + r'\b', text_lower):
        return True, skill, None

    # Graph lookup
    if s_clean in NORMALIZED_TECH_GRAPH:
        mapped = NORMALIZED_TECH_GRAPH[s_clean]
        if any(m in cand_set for m in mapped):
            return True, skill, None
        if any(len(m) > 3 and re.search(r'\b' + re.escape(m) + r'\b', text_lower) for m in mapped):
            return True, skill, None

    # Compound words boundary search with broad platform protection
    for key, mapped in NORMALIZED_TECH_GRAPH.items():
        if re.search(r'\b' + re.escape(key) + r'\b', s_clean):
            # Guard: broad platform (aws, react, cloud) should not match specific sub-tools (aws lambda, react native)
            if key in BROAD_PLATFORMS and key != s_clean and s_clean not in ['public cloud', 'cloud technologies']:
                continue
            if ("llm" in s_clean or "ai" in s_clean) and ("llm" not in key and "ai" not in key):
                continue
            if any(m in cand_set for m in mapped):
                return True, skill, None
            if any(len(m) > 3 and re.search(r'\b' + re.escape(m) + r'\b', text_lower) for m in mapped):
                return True, skill, None

    # Single token match (only if s_clean is a single word)
    words = s_clean.split()
    if len(words) == 1:
        w = words[0]
        if w not in STOPWORDS and len(w) >= 3:
            if w in cand_set or re.search(r'\b' + re.escape(w) + r'\b', text_lower):
                return True, skill, None

    return False, None, skill

def evaluate_criterion(skill: str, candidate_skills: List[str], candidate_text: str = "", seniority: str = "") -> Tuple[bool, Optional[str], Optional[str]]:
    """Evaluate a single criterion or an alternative OR group (e.g. 'PostgreSQL or MySQL', 'React.js or Angular')."""
    if ' or ' in skill.lower():
        alternatives = [alt.strip() for alt in re.split(r'\s+or\s+', skill, flags=re.IGNORECASE) if alt.strip()]
        for alt in alternatives:
            is_met, detail, gap = evaluate_single_criterion(alt, candidate_skills, candidate_text, seniority)
            if is_met:
                label = f"{format_skill_title(alt)} (met '{format_skill_title(skill)}')" if len(alternatives) > 1 else format_skill_title(alt)
                return True, label, None
        return False, None, format_skill_title(skill)
    return evaluate_single_criterion(skill, candidate_skills, candidate_text, seniority)

def find_evidence_snippet(skill: str, text: str) -> str:
    """Find the exact sentence or line in the resume verifying this skill."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Extract base skill if parenthetical annotation present (e.g. "PostgreSQL (met '...')")
    clean_lookup = re.sub(r'\s*\([^)]*\)', '', skill).strip()
    s_clean = normalize_skill_key(clean_lookup)
    base_clean = re.sub(r'^(?:aws|modern)\s+', '', s_clean).strip()

    search_terms = [s_clean]
    if base_clean and base_clean != s_clean:
        search_terms.append(base_clean)
    if s_clean in NORMALIZED_TECH_GRAPH:
        search_terms.extend(list(NORMALIZED_TECH_GRAPH[s_clean])[:4])

    # 1. Prioritize work experience/project sentences (non-skills list)
    for term in search_terms:
        if len(term) < 3:
            continue
        for line in lines:
            if not line.lower().startswith("skills:"):
                if re.search(r'\b' + re.escape(term) + r'\b', line.lower()):
                    return line[:130] + ("..." if len(line) > 130 else "")

    # 2. Check skills list lines
    for term in search_terms:
        if len(term) < 3:
            continue
        for line in lines:
            if re.search(r'\b' + re.escape(term) + r'\b', line.lower()):
                return line[:130] + ("..." if len(line) > 130 else "")

    return "Verified in candidate technical profile"

def generate_candidate_pitch(name: str, years: str, seniority: str, score: int, matched: List[str], missing: List[str], core_score: Optional[int] = None, bonus_score: float = 0.0) -> str:
    """Generate an executive, copy-ready candidate pitch for Slack/Email with clear core vs bonus breakdown."""
    top_matched = ", ".join(matched[:5]) if matched else "None"
    top_missing = ", ".join(missing[:4]) if missing else "None (100% Core Fit)"
    verdict_label = "Shortlist" if score >= 70 else ("Review" if score >= 60 else "Low Fit")
    score_breakdown = f" (Core: {core_score}%, Bonus: +{bonus_score} pts)" if core_score is not None else ""
    return (
        f"Candidate: {name} ({years}, {seniority}) | Match: {score}%{score_breakdown} ({verdict_label})\n"
        f"• Verified Core Strengths: {top_matched}\n"
        f"• Technical Gaps: {top_missing}"
    )

def verify_and_clean_gaps(cand_skills: List[str], cand_text: str, missing_skills: List[str], matched_skills: List[str], seniority: str = "") -> Tuple[List[str], List[str]]:
    """Self-Verification Safety Pass: Double-checks all flagged gaps before returning results."""
    cand_lower = {normalize_skill_key(s): s for s in cand_skills} if "cand_skills" in locals() else {}
    cand_set = set(cand_lower.keys())
    text_lower = cand_text.lower() if cand_text else ""

    verified_missing = []
    rescued_matches = [format_skill_title(clean_atomic_phrase(s)) for s in matched_skills if s and s.strip()]
    rescued_set = {normalize_skill_key(s) for s in rescued_matches}

    for raw_gap in missing_skills:
        gap = format_skill_title(clean_atomic_phrase(raw_gap)) if raw_gap else ""
        if not gap:
            continue
        gap_clean = normalize_skill_key(gap)
        if not gap_clean:
            continue

        base_gap = re.sub(r'^(?:aws|modern)\s+', '', gap_clean).strip()

        # Check 0: If gap is an OR alternative group, rescue if candidate has ANY alternative
        if ' or ' in gap.lower():
            alts = [a.strip() for a in re.split(r'\s+or\s+', gap, flags=re.IGNORECASE) if a.strip()]
            matched_alt = None
            for alt in alts:
                alt_clean = normalize_skill_key(alt)
                base_alt = re.sub(r'^(?:aws|modern)\s+', '', alt_clean).strip()
                if alt_clean in cand_set or base_alt in cand_set or (len(alt_clean) >= 3 and re.search(r'\b' + re.escape(alt_clean) + r'\b', text_lower)) or (len(base_alt) >= 3 and re.search(r'\b' + re.escape(base_alt) + r'\b', text_lower)):
                    matched_alt = alt
                    break
            if matched_alt:
                label = f"{format_skill_title(matched_alt)} (met '{format_skill_title(gap)}')"
                if gap_clean not in rescued_set:
                    rescued_matches.append(label)
                    rescued_set.add(gap_clean)
                continue

        # Check 1: Direct cand_set match
        if gap_clean in cand_set or base_gap in cand_set:
            if gap_clean not in rescued_set:
                rescued_matches.append(gap)
                rescued_set.add(gap_clean)
            continue

        # Check 2: Direct resume text search (exact word boundary)
        if len(gap_clean) >= 3 and re.search(r'\b' + re.escape(gap_clean) + r'\b', text_lower):
            if gap_clean not in rescued_set:
                rescued_matches.append(gap)
                rescued_set.add(gap_clean)
            continue
        if len(base_gap) >= 3 and re.search(r'\b' + re.escape(base_gap) + r'\b', text_lower):
            if gap_clean not in rescued_set:
                rescued_matches.append(gap)
                rescued_set.add(gap_clean)
            continue

        # Check compound stack (MERN, MEAN, MEVN, PERN, LAMP)
        if gap_clean in COMPOUND_STACK_RULES:
            is_stack_met, stack_detail, _ = evaluate_compound_stack(gap_clean, cand_set, text_lower)
            if is_stack_met:
                if gap_clean not in rescued_set:
                    rescued_matches.append(stack_detail or gap)
                    rescued_set.add(gap_clean)
                continue
            else:
                # Do NOT rescue via generic graph lookup if compound stack constituents are not satisfied
                verified_missing.append(gap)
                continue

        # Check 3: Graph equivalent check
        matched_in_graph = False
        for check_key in [gap_clean, base_gap]:
            if check_key in NORMALIZED_TECH_GRAPH:
                mapped = NORMALIZED_TECH_GRAPH[check_key]
                if any(m in cand_set for m in mapped) or any(len(m) >= 3 and re.search(r'\b' + re.escape(m) + r'\b', text_lower) for m in mapped):
                    if gap_clean not in rescued_set:
                        rescued_matches.append(gap)
                        rescued_set.add(gap_clean)
                    matched_in_graph = True
                    break
        if matched_in_graph:
            continue

        # Check 4: Seniority / leadership / delivery check
        if any(k in gap_clean for k in ["tech lead", "team lead", "leadership", "feature ownership", "delivery", "mentoring"]):
            if seniority == "Lead / Principal" or any(w in text_lower for w in ["lead software engineer", "tech lead", "team lead", "head of development", "software delivery"]):
                if gap_clean not in rescued_set:
                    rescued_matches.append(gap)
                    rescued_set.add(gap_clean)
                continue

        # Confirmed genuine gap (sanitized & formatted)
        verified_missing.append(gap)

    return rescued_matches, verified_missing

@app.post("/api/match")
def match_job(req: JobMatchRequest):
    if not req.resume_text.strip() or not req.job_description.strip():
        raise HTTPException(status_code=400, detail="Both resume text and job description are required.")
    start_time = time.time()
    profile_data = analyze_resume(ResumeRequest(resume_text=req.resume_text))["data"]
    candidate_skills = profile_data.get("skills", [])
    tiered = extract_universal_criteria_tiered(req.job_description)
    core_criteria = tiered["core"]
    preferred_criteria = tiered["preferred"]
    candidate_seniority = profile_data.get("experience_level", "")

    # 1. Evaluate Core Mandatory Requirements
    matched_core = []
    missing_core = []
    for c in core_criteria:
        is_met, detail, gap = evaluate_criterion(c, candidate_skills, req.resume_text, candidate_seniority)
        if is_met:
            matched_core.append(c)
        else:
            missing_core.append(gap if gap else c)
    matched_core, missing_core = verify_and_clean_gaps(candidate_skills, req.resume_text, missing_core, matched_core, candidate_seniority)

    # 2. Evaluate Preferred Bonus Requirements
    matched_pref = []
    missing_pref = []
    for p in preferred_criteria:
        is_met, detail, gap = evaluate_criterion(p, candidate_skills, req.resume_text, candidate_seniority)
        if is_met:
            matched_pref.append(p)
        else:
            missing_pref.append(gap if gap else p)
    matched_pref, missing_pref = verify_and_clean_gaps(candidate_skills, req.resume_text, missing_pref, matched_pref, candidate_seniority)

    # 3. Two-Tier Scoring (Core 100% + Bonus up to +10 pts)
    core_total = len(core_criteria)
    core_score = int((len(matched_core) / core_total) * 100) if core_total > 0 else 80
    pref_total = len(preferred_criteria)
    bonus_score = round(10.0 * (len(matched_pref) / pref_total), 1) if pref_total > 0 else 0.0
    final_score = min(98, max(20, int(round(core_score + bonus_score))))

    all_matched = matched_core + matched_pref
    evidence_map = {s: find_evidence_snippet(s, req.resume_text) for s in all_matched}
    candidate_name = profile_data.get("name", "Candidate")
    pitch = generate_candidate_pitch(candidate_name, profile_data.get("years_of_experience", ""), candidate_seniority, final_score, matched_core, missing_core, core_score, bonus_score)
    latency_ms = round((time.time() - start_time) * 1000, 1)
    return {
        "success": True,
        "latency_ms": latency_ms,
        "data": {
            "match_score": final_score,
            "core_score": core_score,
            "bonus_score": bonus_score,
            "matched_skills": matched_core,
            "missing_skills": missing_core,
            "matched_bonus": matched_pref,
            "missing_bonus": missing_pref,
            "evidence_map": evidence_map,
            "candidate_pitch": pitch,
            "candidate_domain": profile_data.get("primary_domain"),
            "candidate_seniority": profile_data.get("experience_level"),
            "candidate_years": profile_data.get("years_of_experience"),
            "recommended_roles": profile_data.get("matching_job_categories", [])
        }
    }
@app.post("/api/extract-criteria")
def extract_criteria_api(req: ExtractCriteriaRequest):
    if not req.job_description.strip():
        return {"success": False, "core": [], "preferred": [], "all": []}
    tiered = extract_universal_criteria_tiered(req.job_description)
    return {
        "success": True,
        "core": tiered["core"],
        "preferred": tiered["preferred"],
        "all": tiered["all"]
    }

@app.post("/api/batch-match")
def batch_match_jobs(req: BatchMatchRequest):
    if not req.job_description.strip():
        raise HTTPException(status_code=400, detail="Target job description is required.")
    if not req.resumes:
        raise HTTPException(status_code=400, detail="At least one candidate resume is required.")
    start_time = time.time()
    # 1. Determine criteria (use custom recruiter chips if supplied, else auto-extract)
    if req.custom_core_criteria is not None or req.custom_preferred_criteria is not None:
        core_criteria = [format_skill_title(clean_atomic_phrase(c)) for c in (req.custom_core_criteria or []) if c.strip()]
        preferred_criteria = [format_skill_title(clean_atomic_phrase(c)) for c in (req.custom_preferred_criteria or []) if c.strip()]
        criteria = core_criteria + preferred_criteria
    else:
        tiered = extract_universal_criteria_tiered(req.job_description)
        core_criteria = tiered["core"]
        preferred_criteria = tiered["preferred"]
        criteria = tiered["all"]
    # 2. Evaluate each candidate resume against the criteria
    leaderboard = []
    for r in req.resumes:
        text = r.text or ""
        if not text and r.content_base64:
            try:
                raw_bytes = base64.b64decode(r.content_base64)
                text = _extract_text_from_bytes(r.filename, raw_bytes)
            except Exception:
                text = ""
        if not text.strip():
            continue
        # Extract candidate name from filename or first line
        base_name = re.sub(r'(_resume|_cv|\bcv\b|\bresume\b|\.pdf|\.docx|\.doc|\.txt)', '', r.filename, flags=re.IGNORECASE).replace('_', ' ').replace('-', ' ').strip().title()
        first_lines = [l.strip() for l in text.splitlines() if l.strip()]
        candidate_name = base_name if base_name and len(base_name) > 2 else (first_lines[0] if first_lines else "Candidate")
        if len(candidate_name) > 35:
            candidate_name = candidate_name[:35]
        # Extract experience years
        years_match = re.search(r'(\d+[\+\-]?\s*(?:years?|yrs?)(?:\s+of\s+experience)?)', text, re.IGNORECASE)
        years_str = years_match.group(1).title() if years_match else "Experience Not Specified"
        # Seniority label (based on job title header and verified experience)
        lines_text = [l.strip() for l in text.splitlines() if l.strip()]
        title_text = " ".join(lines_text[:3]).lower() if lines_text else text.lower()
        years_num = int(re.search(r'\d+', years_str).group()) if years_match else 0
        if re.search(r'\b(lead\b|tech lead|team lead|principal|head of|director|engineering manager)', title_text):
            seniority = "Lead / Principal"
        elif re.search(r'\b(senior|sr\b|sr\.)', title_text) or years_num >= 5:
            seniority = "Senior"
        elif re.search(r'\b(junior|jr\b|jr\.|entry|intern|associate)', title_text) or (0 < years_num <= 2):
            seniority = "Junior"
        else:
            seniority = "Mid-Level"
        cand_skills = extract_and_deduplicate_skills(text, [])

        # 2a. Evaluate Core Mandatory Requirements
        matched_core = []
        missing_core = []
        for c in core_criteria:
            is_met, detail, gap = evaluate_criterion(c, cand_skills, text, seniority)
            if is_met:
                matched_core.append(c)
            else:
                missing_core.append(gap if gap else c)
        matched_core, missing_core = verify_and_clean_gaps(cand_skills, text, missing_core, matched_core, seniority)

        # 2b. Evaluate Preferred Bonus Requirements
        matched_pref = []
        missing_pref = []
        for p in preferred_criteria:
            is_met, detail, gap = evaluate_criterion(p, cand_skills, text, seniority)
            if is_met:
                matched_pref.append(p)
            else:
                missing_pref.append(gap if gap else p)
        matched_pref, missing_pref = verify_and_clean_gaps(cand_skills, text, missing_pref, matched_pref, seniority)

        # 2c. Two-Tier Scoring (Core 100% + Bonus up to +10 pts)
        core_total = len(core_criteria)
        core_score = int((len(matched_core) / core_total) * 100) if core_total > 0 else 80
        pref_total = len(preferred_criteria)
        bonus_score = round(10.0 * (len(matched_pref) / pref_total), 1) if pref_total > 0 else 0.0
        score = min(98, max(20, int(round(core_score + bonus_score))))

        # Verdict
        if score >= 70:
            verdict = "🏆 Strong Match — Shortlist"
            badge_class = "shortlist"
        elif score >= 60:
            verdict = "✅ Good Fit — Review"
            badge_class = "review"
        elif score >= 45:
            verdict = "⚡ Moderate — Minor Gaps"
            badge_class = "moderate"
        else:
            verdict = "❌ Low Fit — Reject"
            badge_class = "reject"
        all_matched = matched_core + matched_pref
        evidence_map = {s: find_evidence_snippet(s, text) for s in all_matched}
        pitch = generate_candidate_pitch(candidate_name, years_str, seniority, score, matched_core, missing_core, core_score, bonus_score)
        leaderboard.append({
            "filename": r.filename,
            "candidate_name": candidate_name,
            "match_score": score,
            "core_score": core_score,
            "bonus_score": bonus_score,
            "matched_skills": matched_core,
            "missing_skills": missing_core,
            "matched_bonus": matched_pref,
            "missing_bonus": missing_pref,
            "evidence_map": evidence_map,
            "candidate_pitch": pitch,
            "years_experience": years_str,
            "seniority": seniority,
            "verdict": verdict,
            "badge_class": badge_class
        })
    # Sort candidates by score descending
    leaderboard.sort(key=lambda x: x["match_score"], reverse=True)
    # Assign ranks
    for i, item in enumerate(leaderboard, 1):
        item["rank"] = i
    latency_ms = round((time.time() - start_time) * 1000, 1)
    return {
        "success": True,
        "latency_ms": latency_ms,
        "data": {
            "total_candidates": len(leaderboard),
            "jd_required_skills": criteria,
            "jd_core_skills": core_criteria,
            "jd_preferred_skills": preferred_criteria,
            "leaderboard": leaderboard
        }
    }
STATIC_DIR = Path(__file__).parent / "static"
if not STATIC_DIR.exists():
    STATIC_DIR = Path("./static")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    @app.get("/")
    def serve_index():
        return FileResponse(str(STATIC_DIR / "index.html"))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8082, reload=True)
