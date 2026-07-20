
JD_PARSE_PROMPT = """You are a strict information-extraction engine. Extract ONLY facts explicitly \
stated in the job description below. Never rate, score, or recommend anything.

JOB DESCRIPTION:
{jd_text}

Return ONLY valid JSON, no markdown, no preamble, no commentary, in exactly this shape:
{{
  "required_skills": ["<skill explicitly required/must-have>"],
  "preferred_skills": ["<skill explicitly preferred/nice-to-have/bonus>"],
  "responsibilities": ["<key duty or responsibility>"],
  "min_experience_years": <number of years required, or null if not stated>,
  "education_level": "<one of: high_school, associate, bachelor, master, phd, or null>",
  "certifications": ["<certification explicitly mentioned>"],
  "keywords": ["<other important domain keyword or tool>"]
}}

Rules:
- Use [] for lists and null for scalars when there is no evidence. Never invent, infer, or assume.
- Each skill/keyword: a short phrase (max 4 words), not a sentence.
- Classify education_level only if the JD explicitly states a degree requirement.
"""


RESUME_PARSE_PROMPT = """You are a strict information-extraction engine. Extract ONLY facts explicitly \
stated in the COMPLETE resume below. Never rate, score, recommend, or judge fit for any job. Never \
infer skills, experience, or education that are not directly stated in the text.

RESUME (candidate: {candidate_name}):
{resume_text}

Return ONLY valid JSON, no markdown, no preamble, no commentary, in exactly this shape:
{{
  "skills": ["<technical or soft skill explicitly listed>"],
  "tools": ["<tool or software explicitly mentioned>"],
  "programming_languages": ["<programming language explicitly mentioned>"],
  "frameworks": ["<framework or library explicitly mentioned>"],
  "databases": ["<database technology explicitly mentioned>"],
  "cloud_platforms": ["<cloud platform explicitly mentioned>"],
  "projects": [
    {{"name": "<project name>", "description": "<1 sentence, only what is stated>", "technologies_used": ["<tech mentioned for this project>"]}}
  ],
  "experience_entries": [
    {{"company": "<company or org, or null>", "role": "<title, or null>", "start_date": "<as stated, e.g. 'Jun 2023', or null>", "end_date": "<as stated, e.g. 'Present', or null>", "duration_text": "<as stated, e.g. '1 year', or null>"}}
  ],
  "education": [
    {{"degree_text": "<original degree text exactly as written, e.g. 'B.Tech in Computer Science'>", "institution": "<institution, or null>", "field": "<field of study, or null>"}}
  ],
  "education_level": "<highest degree normalized to one of: high_school, associate, bachelor, master, phd, or null>",
  "education_field": "<field of study of the highest degree, or null>",
  "certifications": ["<certification explicitly listed>"],
  "achievements": ["<award, honor, or notable achievement explicitly stated>"],
  "ats_keywords": [
    "<important ATS keyword explicitly mentioned such as Agile, Scrum, Leadership, Problem Solving, Communication, CI/CD, Microservices, Docker, Kubernetes>"
  ]
}}

Rules:
- Use [] for lists and null for scalars when there is no evidence. Never invent, infer, or assume.
- Do NOT calculate or estimate total years of experience — extract each experience_entries item as stated; \
duration will be computed separately.
- education_level is your normalization of the candidate's highest stated degree only; if no degree is \
stated, use null.
- Keep each skill/tool/language/framework as a short phrase, not a sentence.
"""