// Gemini API utility
// Uses Gemini 3.5 Flash: inherits Gemini 3 Pro reasoning at Flash speed.
// Strong structured JSON adherence, weighted ATS scoring, and acronym/synonym matching.

const GEMINI_MODEL = "gemini-3.5-flash";
const GEMINI_BASE = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

const RESUME_JSON_SCHEMA = {
  type: "object",
  properties: {
    personal: {
      type: "object",
      properties: {
        firstName: { type: "string" },
        lastName: { type: "string" },
        email: { type: "string" },
        phone: { type: "string" },
        location: { type: "string" },
        linkedin: { type: "string" },
        github: { type: "string" },
        portfolio: { type: "string" },
      },
      required: [
        "firstName",
        "lastName",
        "email",
        "phone",
        "location",
        "linkedin",
        "github",
        "portfolio",
      ],
    },
    summary: { type: "string" },
    experience: {
      type: "array",
      items: {
        type: "object",
        properties: {
          title: { type: "string" },
          company: { type: "string" },
          location: { type: "string" },
          startDate: { type: "string" },
          endDate: { type: "string" },
          current: { type: "boolean" },
          bullets: { type: "array", items: { type: "string" } },
        },
        required: [
          "title",
          "company",
          "location",
          "startDate",
          "endDate",
          "current",
          "bullets",
        ],
      },
    },
    education: {
      type: "array",
      items: {
        type: "object",
        properties: {
          school: { type: "string" },
          degree: { type: "string" },
          location: { type: "string" },
          startDate: { type: "string" },
          endDate: { type: "string" },
          current: { type: "boolean" },
          cgpa: { type: "string" },
          bullets: { type: "array", items: { type: "string" } },
        },
        required: [
          "school",
          "degree",
          "location",
          "startDate",
          "endDate",
          "current",
          "cgpa",
          "bullets",
        ],
      },
    },
    skills: {
      type: "object",
      properties: {
        programmingLanguages: { type: "array", items: { type: "string" } },
        csConcepts: { type: "array", items: { type: "string" } },
        webDevelopment: { type: "array", items: { type: "string" } },
        databases: { type: "array", items: { type: "string" } },
        cloudPlatforms: { type: "array", items: { type: "string" } },
        mlAI: { type: "array", items: { type: "string" } },
        mobileDevelopment: { type: "array", items: { type: "string" } },
        custom: {
          type: "array",
          items: {
            type: "object",
            properties: {
              title: { type: "string" },
              items: { type: "array", items: { type: "string" } },
            },
            required: ["title", "items"],
          },
        },
      },
      required: [
        "programmingLanguages",
        "csConcepts",
        "webDevelopment",
        "databases",
        "cloudPlatforms",
        "mlAI",
        "mobileDevelopment",
      ],
    },
    projects: {
      type: "array",
      items: {
        type: "object",
        properties: {
          name: { type: "string" },
          organization: { type: "string" },
          link: { type: "string" },
          location: { type: "string" },
          startDate: { type: "string" },
          endDate: { type: "string" },
          current: { type: "boolean" },
          bullets: { type: "array", items: { type: "string" } },
        },
        required: [
          "name",
          "organization",
          "link",
          "location",
          "startDate",
          "endDate",
          "current",
          "bullets",
        ],
      },
    },
    certifications: { type: "array", items: { type: "string" } },
    achievements: { type: "array", items: { type: "string" } },
    languages: {
      type: "array",
      items: {
        type: "object",
        properties: {
          name: { type: "string" },
          proficiency: { type: "string" },
        },
        required: ["name", "proficiency"],
      },
    },
    publications: { type: "array", items: { type: "string" } },
  },
  required: [
    "personal",
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "achievements",
    "languages",
    "publications",
  ],
};

const ATS_JSON_SCHEMA = {
  type: "object",
  properties: {
    score: { type: "integer" },
    grade: { type: "string" },
    title_match: { type: "string" },
    matched_keywords: { type: "array", items: { type: "string" } },
    missing_keywords: { type: "array", items: { type: "string" } },
    must_have_matched: { type: "array", items: { type: "string" } },
    must_have_missing: { type: "array", items: { type: "string" } },
    nice_to_have_matched: { type: "array", items: { type: "string" } },
    nice_to_have_missing: { type: "array", items: { type: "string" } },
    weak_matches: { type: "array", items: { type: "string" } },
    suggestions: { type: "string" },
  },
  required: [
    "score",
    "grade",
    "matched_keywords",
    "missing_keywords",
    "suggestions",
  ],
};

// Pull the visible answer text out of a Gemini response (skipping "thought" parts)
function extractGeminiText(data) {
  const parts = data?.candidates?.[0]?.content?.parts ?? [];
  return parts
    .filter((p) => !p.thought)
    .map((p) => p.text ?? "")
    .join("")
    .trim();
}

async function callGemini(prompt, apiKey, config = {}) {
  const response = await fetch(GEMINI_BASE, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-goog-api-key": apiKey,
    },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: config.temperature ?? 0.7,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: config.maxOutputTokens ?? 2048,
        ...(config.responseMimeType
          ? { responseMimeType: config.responseMimeType }
          : {}),
        ...(config.responseJsonSchema
          ? { responseJsonSchema: config.responseJsonSchema }
          : {}),
        ...(config.thinkingBudget !== undefined
          ? { thinkingConfig: { thinkingBudget: config.thinkingBudget } }
          : {}),
      },
    }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    const msg = err?.error?.message || `HTTP ${response.status}`;
    throw new Error(`Gemini API Error: ${msg}`);
  }

  const data = await response.json();
  const text = extractGeminiText(data);
  if (!text) throw new Error("Gemini returned empty response.");
  return text;
}

/**
 * Multi-turn variant of callGemini.
 *
 * Sends a stable `systemInstruction` (kept identical across a chat session so
 * Gemini's automatic implicit caching discounts the repeated tokens) plus the
 * rolling conversation as a `contents` array of { role, parts } turns.
 */
async function callGeminiConversation(
  { systemInstruction, contents },
  apiKey,
  config = {},
) {
  const body = {
    contents,
    generationConfig: {
      temperature: config.temperature ?? 0.7,
      topK: 40,
      topP: 0.95,
      maxOutputTokens: config.maxOutputTokens ?? 1024,
      ...(config.thinkingBudget !== undefined
        ? { thinkingConfig: { thinkingBudget: config.thinkingBudget } }
        : {}),
    },
  };
  if (systemInstruction) {
    body.systemInstruction = { parts: [{ text: systemInstruction }] };
  }

  const response = await fetch(GEMINI_BASE, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-goog-api-key": apiKey,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    const msg = err?.error?.message || `HTTP ${response.status}`;
    throw new Error(`Gemini API Error: ${msg}`);
  }

  const data = await response.json();
  const text = extractGeminiText(data);
  if (!text) throw new Error("Gemini returned empty response.");
  return text;
}

function parseJsonSafely(rawText, fallbackMessage) {
  const cleaned = String(rawText || "")
    .replace(/```json\s*/gi, "")
    .replace(/```\s*/gi, "")
    .replace(/^\uFEFF/, "")
    .trim();

  const candidates = [];

  if (cleaned) candidates.push(cleaned);
  if (cleaned) {
    candidates.push(
      cleaned
        .replace(/,\s*([}\]])/g, "$1")
        .replace(/[\u0000-\u0019]+/g, " ")
        .trim(),
    );
  }

  const match = cleaned.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
  if (match) {
    candidates.push(match[0]);
    candidates.push(
      match[0]
        .replace(/,\s*([}\]])/g, "$1")
        .replace(/[\u0000-\u0019]+/g, " ")
        .trim(),
    );
  }

  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      return JSON.parse(candidate);
    } catch (_) {
      // Try the next candidate.
    }
  }

  throw new Error(fallbackMessage);
}

function normalizeStringArray(value) {
  if (Array.isArray(value)) {
    return value.map((item) => String(item ?? "").trim()).filter(Boolean);
  }
  if (typeof value === "string") {
    return value
      .split(/[\n,]/)
      .map((item) => item.replace(/^[\-\*\u2022]\s*/, "").trim())
      .filter(Boolean);
  }
  return [];
}

function normalizeATSResult(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  const scoreValue = Number(
    source.score ?? source.ats_score ?? source.match_score ?? 0,
  );

  return {
    score: Number.isFinite(scoreValue)
      ? Math.max(0, Math.min(100, Math.round(scoreValue)))
      : 0,
    grade: String(source.grade ?? source.rating ?? "Fair").trim() || "Fair",
    title_match: String(source.title_match ?? source.titleMatch ?? "").trim(),
    matched_keywords: normalizeStringArray(
      source.matched_keywords ?? source.matchedKeywords ?? source.matched,
    ),
    missing_keywords: normalizeStringArray(
      source.missing_keywords ?? source.missingKeywords ?? source.missing,
    ),
    must_have_matched: normalizeStringArray(
      source.must_have_matched ?? source.mustHaveMatched,
    ),
    must_have_missing: normalizeStringArray(
      source.must_have_missing ?? source.mustHaveMissing,
    ),
    nice_to_have_matched: normalizeStringArray(
      source.nice_to_have_matched ?? source.niceToHaveMatched,
    ),
    nice_to_have_missing: normalizeStringArray(
      source.nice_to_have_missing ?? source.niceToHaveMissing,
    ),
    weak_matches: normalizeStringArray(
      source.weak_matches ?? source.weakMatches,
    ),
    suggestions: String(
      source.suggestions ?? source.recommendations ?? source.feedback ?? "",
    ).trim(),
  };
}

function parseLabeledATSResponse(text) {
  const raw = String(text || "").trim();
  const scoreMatch = raw.match(/^\s*score\s*:\s*(\d{1,3})/im);
  const gradeMatch = raw.match(/^\s*grade\s*:\s*([^\n\r]+)/im);
  const titleMatchMatch = raw.match(/^\s*title[_\s-]*match\s*:\s*([^\n\r]+)/im);
  const matchedMatch = raw.match(
    /^\s*matched[_\s-]*keywords?\s*:\s*([^\n\r]*)/im,
  );
  const missingMatch = raw.match(
    /^\s*missing[_\s-]*keywords?\s*:\s*([^\n\r]*)/im,
  );
  const mustMatchedMatch = raw.match(
    /^\s*must[_\s-]*have[_\s-]*matched\s*:\s*([^\n\r]*)/im,
  );
  const mustMissingMatch = raw.match(
    /^\s*must[_\s-]*have[_\s-]*missing\s*:\s*([^\n\r]*)/im,
  );
  const niceMatchedMatch = raw.match(
    /^\s*nice[_\s-]*to[_\s-]*have[_\s-]*matched\s*:\s*([^\n\r]*)/im,
  );
  const niceMissingMatch = raw.match(
    /^\s*nice[_\s-]*to[_\s-]*have[_\s-]*missing\s*:\s*([^\n\r]*)/im,
  );
  const weakMatchesMatch = raw.match(
    /^\s*weak[_\s-]*matches?\s*:\s*([^\n\r]*)/im,
  );
  const suggestionsMatch = raw.match(/^\s*suggestions?\s*:\s*([\s\S]*)$/im);

  const result = normalizeATSResult({
    score: scoreMatch ? Number(scoreMatch[1]) : 0,
    grade: gradeMatch ? gradeMatch[1].trim() : "Fair",
    title_match: titleMatchMatch ? titleMatchMatch[1].trim() : "",
    matched_keywords: matchedMatch ? matchedMatch[1].trim() : [],
    missing_keywords: missingMatch ? missingMatch[1].trim() : [],
    must_have_matched: mustMatchedMatch ? mustMatchedMatch[1].trim() : [],
    must_have_missing: mustMissingMatch ? mustMissingMatch[1].trim() : [],
    nice_to_have_matched: niceMatchedMatch ? niceMatchedMatch[1].trim() : [],
    nice_to_have_missing: niceMissingMatch ? niceMissingMatch[1].trim() : [],
    weak_matches: weakMatchesMatch ? weakMatchesMatch[1].trim() : [],
    suggestions: suggestionsMatch ? suggestionsMatch[1].trim() : "",
  });

  if (result.score === 0 && result.matched_keywords.length > 0) {
    const total =
      result.matched_keywords.length + result.missing_keywords.length;
    result.score =
      total > 0
        ? Math.round((result.matched_keywords.length / total) * 100)
        : 0;
    if (result.score >= 80) result.grade = "Excellent";
    else if (result.score >= 60) result.grade = "Good";
    else if (result.score >= 40) result.grade = "Fair";
    else result.grade = "Poor";
  }

  if (!result.suggestions) {
    result.suggestions =
      "Review the missing keywords and update your summary and bullet points to align more closely with the job description.";
  }

  return result;
}

/**
 * Tailor a structured resume for a specific job and return structured JSON
 */
async function tailorResumeStructured(
  { jobTitle, company, jobDescription, resumeData, candidateName },
  apiKey,
) {
  const prompt = `You are a senior resume writer and ATS (Applicant Tracking System) optimization specialist with 15+ years of experience getting candidates shortlisted by recruiters at Fortune 500 companies and top tech firms.

Your job is to transform a structured resume into a highly optimized, job-specific, ATS-friendly version while strictly preserving truthfulness.

You behave as TWO experts at once:
1) An ATS keyword-matching engine that does literal string matching on JD terms
2) A human recruiter who skims for clarity, role fit, and measurable impact in 6 seconds

---

CANDIDATE NAME: ${candidateName || "Candidate"}

JOB DETAILS:
Title: ${jobTitle || "Not specified"}
Company: ${company || "Not specified"}
Description:
${jobDescription}

CURRENT STRUCTURED RESUME JSON:
${JSON.stringify(resumeData, null, 2)}

---

PRIMARY OBJECTIVE:
Return a job-tailored version of the resume in the EXACT SAME JSON SCHEMA, optimized for:
- ATS keyword matching (literal + acronym variants)
- Role and title relevance
- Recruiter readability and 6-second scan
- Conciseness, clarity, and measurable impact

Match the LANGUAGE of the job description (write English content for English JDs, etc.).

---

STRICT RULES (NON-NEGOTIABLE):
1. Do NOT invent or assume any new experience, companies, roles, dates, projects, certifications, or metrics.
2. Do NOT exaggerate skills beyond what is already stated. "Concepts" or "basic" stays "concepts" — never upgraded to advanced.
3. Do NOT fabricate impact metrics (percentages, dollar values, user counts). If the original lacked them, do not invent.
4. Only rewrite, reorder, and improve wording of existing content.
5. Preserve all links, company names, dates, and project names EXACTLY as provided.
6. Maintain the exact JSON structure and field names (programmingLanguages, csConcepts, webDevelopment, databases, cloudPlatforms, mlAI, mobileDevelopment).
7. Bullet points must remain arrays of strings. No nested objects, no markdown.
8. If a section is empty in input, return it empty in output.
9. Output ONLY valid JSON. No explanations, no markdown fences, no extra text.

---

OPTIMIZATION LOGIC:

1) ROLE DETECTION:
- Analyze the JD to determine primary role focus (Backend, Frontend, Full Stack, AI/ML, Mobile, DevOps, Data, SRE, QA, Product, etc.).
- Adapt resume emphasis dynamically. Do NOT assume any fixed stack.

2) KEYWORD EXTRACTION & CLASSIFICATION:
- Extract critical terms from the JD: technologies, tools, frameworks, languages, methodologies, concepts (REST APIs, Microservices, OOP, CI/CD, Agile, TDD), certifications, and domain terms.
- Classify each keyword as:
  a. MUST-HAVE: appears with "required", "must have", "minimum", "X+ years", or in core responsibilities.
  b. NICE-TO-HAVE: appears with "preferred", "plus", "bonus", "nice to have".
- GUARANTEE: 100% of MUST-HAVE keywords that the candidate truthfully possesses MUST appear in Summary + Skills + at least one Experience or Project bullet.

3) ACRONYM & FULL-FORM HANDLING (CRITICAL FOR ATS):
- If the JD uses an acronym (e.g., "AWS", "ML", "K8s") AND its full form ("Amazon Web Services", "Machine Learning", "Kubernetes") — include BOTH forms at least once across the resume, ideally as "AWS (Amazon Web Services)" on first mention.
- If the JD uses only the acronym, use the acronym. If only the full form, use both.
- This is non-negotiable: ATS systems do literal string matching and miss acronym-only resumes.

4) EXACT-PHRASE MIRRORING:
- For the top 5–7 most critical keywords/phrases from the JD, mirror the EXACT phrasing verbatim. Do NOT paraphrase.
  Example: JD says "CI/CD pipelines" → write "CI/CD pipelines", NOT "continuous integration workflows".
  Example: JD says "RESTful APIs" → write "RESTful APIs", NOT "REST endpoints".

5) SUMMARY OPTIMIZATION:
- Open with a line containing the EXACT job title from the JD (e.g., "Backend Engineer with 3+ years of experience…"). Title match is the #1 ATS signal.
- Mention 3–5 of the most critical JD technologies/skills the candidate truthfully has.
- Keep to 2–4 lines max. Active voice. No first-person pronouns ("I", "my").
- No clichés: NEVER use "results-driven", "passionate", "detail-oriented", "team player", "hard-working", "go-getter", "self-starter".

6) EXPERIENCE OPTIMIZATION:
- Reorder bullets by relevance to the JD (most relevant first).
- Rewrite as: Action Verb + What + How (tech/method) + Impact (metric if available, qualitative outcome otherwise).
- Use past tense for past roles, present tense for current role. Active voice always.
- If a bullet has no metric, KEEP it but lead with a strong outcome verb (Architected, Eliminated, Streamlined, Reduced, Accelerated, Shipped, Owned, Migrated).
- BANNED PHRASES (never output these): "Responsible for", "Worked on", "Involved in", "Helped with", "Assisted in", "Duties included", "Tasked with", "In charge of".

7) PROJECTS OPTIMIZATION:
- SELECT only the 2–3 most relevant projects from the full list based on JD fit.
- If candidate has fewer than 2, include all. If more than 3, EXCLUDE the least relevant — do NOT shrink-and-include.
- Selection priority: (1) tech-stack overlap with JD, (2) measurable impact/scale, (3) complexity, (4) recency.
- Reorder bullets within each kept project to lead with JD-matching tech and outcomes.

8) SKILLS OPTIMIZATION:
- FILTER: include only skills that are relevant or adjacent to the JD.
- KEEP a skill if it:
  a. Is explicitly mentioned in the JD, OR
  b. Is in the SAME family/ecosystem as a required skill (keep TypeScript for a JavaScript role; keep PostgreSQL for a MySQL role; keep Vue for a React role), OR
  c. Is a foundational tool any engineer benefits from (Git, Linux, OOP, Data Structures, Algorithms).
- REMOVE a skill if it:
  a. Belongs to a completely different domain with no adjacency (mobile-only skills for a pure backend job), OR
  b. Would dilute the candidate's apparent focus (Flutter on a DevOps resume).
- Reorder within each category by JD relevance (most relevant first).
- If a category becomes empty after filtering, return it as an empty array. NEVER invent skills.
- Note: cloudPlatforms category includes Git, Docker, Kubernetes, Linux, AWS/Azure/GCP — keep this grouping.
- CUSTOM CATEGORIES (skills.custom): apply the same KEEP/REMOVE rules. Preserve each custom title verbatim — never rename or merge custom categories. If a custom category's items all get filtered out, remove that custom category entry entirely from the array (do NOT keep an empty one). If skills.custom was empty in input, return an empty array.

9) KEYWORD DENSITY (ANTI-STUFFING):
- No single keyword should appear MORE than 3 times across the entire resume.
- Distribute keywords naturally across Summary, Skills, Experience, and Projects — not stacked in one section.

10) ATS FORMATTING:
- Plain text in all string fields. No emojis, no Unicode bullets, no markdown, no tables.
- Standard punctuation only.

---

BULLET POINT LIMITS (STRICT):
- Each experience entry: MAX 4 bullets
- Each project: MAX 3 bullets
- Summary: 2–4 lines
- Education bullets: MAX 2 per entry
- Do NOT exceed these limits.

BULLET QUALITY STANDARD:
- Format: Action Verb + What + How + Impact
- Strong verbs: Architected, Built, Designed, Developed, Engineered, Implemented, Launched, Migrated, Optimized, Reduced, Refactored, Scaled, Shipped, Streamlined.
- Each bullet ≤ 25 words. One sentence per bullet.

---

FINAL CHECK BEFORE RETURNING:
- Job title appears in summary? ✓
- All truthfully-possessed MUST-HAVE keywords present? ✓
- Acronym + full-form pairs included? ✓
- No banned phrases? ✓
- No keyword exceeds 3 uses? ✓
- Bullet limits respected? ✓
- Valid JSON, matching schema exactly? ✓

GOAL:
Maximize ATS match score AND recruiter shortlisting probability while remaining completely truthful.`;

  const raw = await callGemini(prompt, apiKey, {
    temperature: 0.2,
    maxOutputTokens: 8192,
    responseMimeType: "application/json",
    responseJsonSchema: RESUME_JSON_SCHEMA,
  });

  return parseJsonSafely(
    raw,
    "Could not parse tailored resume structure. Please try again.",
  );
}

/**
 * Generate personalized cover letter
 */
async function generateCoverLetter(
  { jobTitle, company, jobDescription, baseResume, candidateName, email },
  apiKey,
) {
  const today = new Date().toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });

  const companyName = company || "[Company Name]";
  const greeting = company
    ? `Dear ${company} Hiring Team,`
    : "Dear Hiring Manager,";

  const prompt = `You are a senior career coach and professional cover-letter writer. You have placed candidates at top tech companies, Fortune 500 firms, startups, and research labs. You write cover letters that get read in 30 seconds and remembered.

CANDIDATE: ${candidateName || "Candidate"} | ${email || ""}

JOB DETAILS:
Title: ${jobTitle || "this role"}
Company: ${company || "this company"}
Description:
${jobDescription}

CANDIDATE'S BACKGROUND (from resume — TREAT AS GROUND TRUTH, never invent beyond this):
${baseResume}

TASK: Write a compelling, personalized cover letter tailored to THIS job at THIS company.

---

STRICT TRUTHFULNESS RULES:
- Only reference skills, technologies, achievements, and experiences EXPLICITLY present in the candidate background above.
- NEVER claim skills the candidate doesn't have, even if the JD requires them.
- NEVER invent metrics, dates, projects, or company names.
- If the candidate lacks a JD requirement, simply do NOT mention that gap — pivot to adjacent strengths.

---

STRUCTURE (4 paragraphs, ≤ 320 words total):

Paragraph 1 — HOOK (2–3 sentences):
- Open with a specific hook tied to the role or the company's mission/product/recent work — NOT a generic statement.
- BANNED openings: "I am writing to apply", "I am excited to apply", "I am reaching out", "Please accept my application", "My name is".
- Name the exact job title from the JD and the company by name.

Paragraph 2 — WHY YOU (3–4 sentences):
- Cite 2 specific, quantifiable achievements from the candidate background that map directly to JD requirements.
- Lead each with a strong verb (Built, Shipped, Reduced, Scaled, Architected, Led).
- Mirror 4–6 of the top keywords/skills from the JD verbatim — some ATS (Workday, Greenhouse) scan cover letters too.

Paragraph 3 — WHY THIS COMPANY (2–3 sentences):
- Reference something specific about the company — their mission, product, technology stack, recent news, or values implied in the JD.
- Connect candidate's interests/experience to that specific signal. Avoid generic flattery ("I admire your innovative culture").

Paragraph 4 — CALL TO ACTION (1–2 sentences):
- Confident and specific: e.g., "I would welcome a conversation about how my experience with X can support your Y initiative."
- NOT generic: avoid "I look forward to hearing from you", "Thank you for your consideration" as a STANDALONE.

---

TONE CALIBRATION (match the JD's industry tone):
- Finance / Legal / Healthcare / Government → formal, precise
- Startup / Creative / Consumer tech → conversational, warm, but professional
- Enterprise / B2B SaaS → balanced and businesslike

---

BANNED CLICHÉS (NEVER use these — automatic rejection signal to recruiters):
"team player", "passionate", "hard-working", "detail-oriented", "results-driven", "go-getter", "self-starter", "out-of-the-box thinker", "synergy", "go above and beyond", "I believe", "I feel", "rock star", "ninja", "guru".

STYLE RULES:
- Active voice, varied sentence length.
- No contractions in formal-tone letters (use "I am" not "I'm" for finance/legal). Contractions OK for startup tone.
- No exclamation points.
- No first-paragraph mention of salary, relocation, or visa status.

---

EXACT FORMAT TO RETURN (no markdown, no fences, no commentary):

${today}

Hiring Team
${companyName}

${greeting}

[Paragraph 1 — hook]

[Paragraph 2 — why you, with mirrored JD keywords]

[Paragraph 3 — why this company]

[Paragraph 4 — confident, specific CTA]

Best regards,
${candidateName || "[Your Name]"}
${email || "[Email]"}

Return ONLY the letter exactly in this format. No explanations.`;

  return await callGemini(prompt, apiKey, {
    temperature: 0.4,
    maxOutputTokens: 8192,
  });
}

/**
 * Parse raw resume text into structured JSON fields
 */
async function parseResumeStructured(resumeText, apiKey) {
  const prompt = `You are an expert resume parser. Extract ALL information from the resume text below into the required structured resume JSON. Be exhaustive — do not drop any entries.

RESUME TEXT:
${resumeText}

---

STRICT RULES:

1) COMPLETENESS:
- Extract every job, every education entry, every project, every certification, every achievement, every language, every publication.
- If a candidate held MULTIPLE TITLES at the same company, create SEPARATE experience entries for each title — do NOT merge them.
- If a candidate has multiple degrees from the same institution, create SEPARATE education entries.

2) BULLET CLEANUP:
- Strip leading bullet glyphs and whitespace: •, –, —, *, →, ▪, ▫, ●, ○, ◦, ►, ▸, ►, ▶, ‣, ⁃, ·, and ASCII dashes/asterisks.
- Convert ALL-CAPS bullets to sentence case (preserve proper nouns, acronyms, and tech names: AWS, REST, SQL, React, Node.js stay as-is).
- Fix obvious typos and missing periods. Do NOT change meaning or invent details.
- Bullets are arrays of clean plain strings — no nested formatting, no markdown.
- If a bullet wraps across multiple lines in input, join into a single string with a single space.

3) SKILL CATEGORIZATION:
Place each skill in the SINGLE best-fitting category. If a skill plausibly fits two, pick the most specific.
- programmingLanguages: Python, Java, C, C++, C#, JavaScript, TypeScript, Go, Rust, Ruby, PHP, Swift, Kotlin, Scala, R, MATLAB, Bash, Shell, SQL (the language itself).
- csConcepts: Data Structures, Algorithms, OOP, Functional Programming, Operating Systems, DBMS, Computer Networks, System Design, Compilers, Distributed Systems, Concurrency, Multithreading.
- webDevelopment: React, Next.js, Vue, Angular, Svelte, Node.js, Express, NestJS, Django, Flask, FastAPI, Spring, Spring Boot, .NET, Rails, HTML, CSS, Tailwind, Bootstrap, REST APIs, GraphQL, WebSockets.
- databases: MySQL, PostgreSQL, MongoDB, Redis, Cassandra, DynamoDB, Firebase, SQLite, Oracle, MS SQL Server, Elasticsearch, Neo4j.
- cloudPlatforms: AWS, Azure, GCP, Docker, Kubernetes, Terraform, Ansible, Jenkins, GitHub Actions, GitLab CI, CircleCI, Linux, Unix, Git, Nginx, Apache, CI/CD.
- mlAI: TensorFlow, PyTorch, Keras, scikit-learn, Hugging Face, NLP, Computer Vision, LLMs, Prompt Engineering, LangChain, RAG, OpenCV, Pandas, NumPy, Matplotlib.
- mobileDevelopment: React Native, Flutter, Android, iOS, Swift, SwiftUI, Kotlin, Jetpack Compose, Xamarin.
- CUSTOM CATEGORIES: if the resume has its own labeled skill group that doesn't fit any of the 7 fixed buckets (e.g., "Game Development", "Cybersecurity", "Embedded Systems", "Cloud Security", "Hardware", "Design Tools", "Quantum Computing"), preserve it as a custom category in skills.custom: [{ title: "<exact label from resume>", items: [...] }]. Keep the user's original section title verbatim.
- FALLBACK (only if no custom-category label is given): if a single loose skill doesn't fit any of the 7 fixed categories AND has no custom heading, place it in csConcepts. NEVER drop a skill.

4) DATE NORMALIZATION:
- Preferred format: "MMM YYYY" (e.g., "Jan 2022", "Sep 2024").
- If only year is given ("2024"): use "Jan 2024" as startDate and "Dec 2024" as endDate.
- If range like "2022–2024": startDate "Jan 2022", endDate "Dec 2024".
- If currently active ("Present", "Current", "Now", no end given for current role): set current: true and leave endDate as "".
- If dates absent entirely, use "" for both.

5) URL NORMALIZATION:
- linkedin: store the full canonical URL with https:// prefix. If only a username/handle ("john-doe") is given, output "https://linkedin.com/in/john-doe".
- github: same pattern. Bare username → "https://github.com/username".
- portfolio: the candidate's personal website/portfolio. Store the full URL with https:// prefix. If a bare domain ("johndoe.com") is given, prepend "https://". If no portfolio or personal website is present, use "".
- project links: keep as full URLs with https:// prefix. If "github.com/user/repo" given, prepend "https://".
- email: lowercase, trimmed, no surrounding text.
- phone: keep digits, +, -, parentheses, spaces. Strip prose like "Phone:" prefix.

6) LANGUAGE PROFICIENCY:
- Use ONLY one of: "Native", "Professional", "Basic".
- Map common phrases:
  - "Native", "Mother tongue", "Fluent" → "Native"
  - "Professional", "Business", "Advanced", "Working proficiency", "C1", "C2", "B2" → "Professional"
  - "Basic", "Beginner", "Elementary", "Conversational", "A1", "A2", "B1" → "Basic"

7) LOCATION:
- For personal.location and entry locations: use "City, State/Country" format when possible (e.g., "Bangalore, India", "San Francisco, CA").
- If only city is given, output just the city. If only country, output just the country.
- Do not invent locations.

8) MISSING FIELDS:
- Strings missing: use "".
- Arrays missing: use [].
- Booleans missing: use false.
- NEVER omit a required schema field.

9) UNCONVENTIONAL CONTENT:
- Volunteer work / extracurriculars: place in experience if role-like (with title/org/dates), otherwise in achievements.
- Hobbies, interests, personal projects without dates: skip unless they fit projects with a clear name + bullets.
- Workshops/courses (not full certifications): place in certifications.

10) OUTPUT:
- Return ONLY valid JSON matching the schema. No markdown fences, no commentary.`;

  const raw = await callGemini(prompt, apiKey, {
    temperature: 0.1,
    maxOutputTokens: 8192,
    responseMimeType: "application/json",
    responseJsonSchema: RESUME_JSON_SCHEMA,
  });

  return parseJsonSafely(
    raw,
    "Could not parse resume structure. Please try again.",
  );
}

/**
 * Check ATS score and return parsed JSON
 */
async function checkATSScore({ jobDescription, resume }, apiKey) {
  const prompt = `You are a senior ATS (Applicant Tracking System) analyst with 10+ years of experience auditing resumes against systems like Workday, Greenhouse, Lever, iCIMS, and Taleo. You score resumes the way real ATS systems do: literal keyword matching, weighted by importance, with title-match bonuses.

JOB DESCRIPTION:
${jobDescription}

CANDIDATE RESUME:
${resume}

---

INSTRUCTIONS — follow these steps in order:

STEP 1 — KEYWORD EXTRACTION:
Extract important keywords from the JOB DESCRIPTION:
- Hard skills: technologies, tools, frameworks, programming languages, platforms.
- Methodologies: Agile, Scrum, TDD, CI/CD, Microservices, REST, GraphQL.
- Certifications: AWS Certified, PMP, CISSP, etc.
- Domain terms: healthcare, fintech, e-commerce, etc.
- Years-of-experience requirements.

Target keyword count proportional to JD length: ~1 keyword per 30 words of JD, MINIMUM 10, MAXIMUM 40. Do NOT extract generic English words ("collaborate", "communicate", "team") unless explicitly listed as a requirement.

STEP 2 — KEYWORD CLASSIFICATION (MUST-HAVE vs NICE-TO-HAVE):
Tag each extracted keyword:
- MUST-HAVE: appears alongside "required", "must have", "minimum", "X+ years", or in the core responsibilities/qualifications block.
- NICE-TO-HAVE: appears alongside "preferred", "plus", "bonus", "nice to have", "ideally", "a plus".

STEP 3 — RESUME MATCHING (with precise synonym rules):

A keyword is MATCHED if the resume contains ANY of:
(a) The EXACT term (case-insensitive).
(b) The standard acronym ↔ full-form pair:
    - AWS ↔ Amazon Web Services
    - GCP ↔ Google Cloud Platform
    - ML ↔ Machine Learning
    - AI ↔ Artificial Intelligence
    - NLP ↔ Natural Language Processing
    - CV ↔ Computer Vision
    - K8s ↔ Kubernetes
    - JS ↔ JavaScript
    - TS ↔ TypeScript
    - DB ↔ Database
    - API ↔ Application Programming Interface
    - SDK ↔ Software Development Kit
    - CI/CD ↔ Continuous Integration/Continuous Deployment
    - OOP ↔ Object-Oriented Programming
    - REST ↔ RESTful
(c) A documented version variant (React 18 matches "React"; Python 3 matches "Python"; Node.js matches "Node").

A keyword is NOT matched by:
- Parent-of-child (SQL does NOT match MySQL/PostgreSQL specifically; the JD asking "MySQL" needs "MySQL" not just "SQL").
- Language-similar but distinct (Java ≠ JavaScript; Go ≠ Golang is OK because they ARE the same).
- Vague proximity (mentioning "cloud" doesn't match "AWS").

STEP 4 — CONTEXT STRENGTH ASSESSMENT:
For each MATCHED keyword, classify by where it appears:
- STRONG match: keyword appears in Experience or Projects bullets (demonstrated use).
- WEAK match: keyword appears ONLY in the Skills section (claimed but not demonstrated).
Populate weak_matches with the list of weak matches.

STEP 5 — TITLE MATCH:
Compare the candidate's most recent or most relevant job title against the JD title:
- EXACT match (e.g., "Backend Engineer" ↔ "Backend Engineer"): title_match = "exact", bonus = +5
- CLOSE match (e.g., "Software Engineer" ↔ "Backend Engineer", "ML Engineer" ↔ "Machine Learning Engineer"): title_match = "close", bonus = +2
- DIFFERENT (e.g., "Frontend Engineer" ↔ "Backend Engineer", "Marketing Manager" ↔ "Software Engineer"): title_match = "different", bonus = 0
- NONE (no professional title in resume): title_match = "none", bonus = 0

STEP 6 — WEIGHTED SCORE CALCULATION:
Compute the weighted score:
  must_have_total = count of MUST-HAVE keywords
  nice_to_have_total = count of NICE-TO-HAVE keywords
  must_have_matched_count = count of MUST-HAVE matched
  nice_to_have_matched_count = count of NICE-TO-HAVE matched
  weak_match_count = count of weak matches across all matched

  numerator = (must_have_matched_count * 2) + (nice_to_have_matched_count * 1) - (weak_match_count * 0.25)
  denominator = (must_have_total * 2) + (nice_to_have_total * 1)
  base_score = round((numerator / denominator) * 100)

  final_score = min(100, max(0, base_score + title_bonus))

If denominator is 0 (no keywords extracted), set score to 0 and explain in suggestions.

STEP 7 — GRADE:
Assign grade from final_score:
- 80–100 → "Excellent"
- 60–79  → "Good"
- 40–59  → "Fair"
- 0–39   → "Poor"

STEP 8 — STRUCTURED SUGGESTIONS:
Write the "suggestions" field as exactly 3 sentences (or 3 bullets joined with newlines):
1. KEYWORDS: "Add these missing must-have keywords to [section]: <up to 3 keywords>." Reference the exact missing must-haves.
2. PHRASING: One specific rewording or context-strengthening suggestion (e.g., "Move 'AWS' from your Skills section into an Experience bullet to convert a weak match to a strong match.").
3. STRUCTURE: One structural fix (e.g., "Add the exact job title 'Backend Engineer' to your summary line." OR "Include both 'AWS' and 'Amazon Web Services' on first mention for ATS acronym matching.").

---

OUTPUT — return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{
  "score": <integer 0-100, the final_score>,
  "grade": "<Excellent|Good|Fair|Poor>",
  "title_match": "<exact|close|different|none>",
  "matched_keywords": ["<all matched keywords>"],
  "missing_keywords": ["<all missing keywords>"],
  "must_have_matched": ["<must-have keywords found>"],
  "must_have_missing": ["<must-have keywords missing>"],
  "nice_to_have_matched": ["<nice-to-have keywords found>"],
  "nice_to_have_missing": ["<nice-to-have keywords missing>"],
  "weak_matches": ["<matched keywords found only in skills section>"],
  "suggestions": "<3-sentence structured suggestions per Step 8>"
}

Rules:
- "missing_keywords" must never be empty if any JD keywords were not matched.
- "score" must be a real calculated integer per Step 6. Never default to 0 unless denominator is 0.
- All keyword arrays must be flat string arrays.`;

  try {
    const raw = await callGemini(prompt, apiKey, {
      temperature: 0.1,
      maxOutputTokens: 3072,
      thinkingBudget: 0,
      responseMimeType: "application/json",
      responseJsonSchema: ATS_JSON_SCHEMA,
    });

    const parsed = normalizeATSResult(
      parseJsonSafely(
        raw,
        "Could not parse ATS score response. Please try again.",
      ),
    );
    // If score came back as 0 but we have matched keywords, recalculate
    if (parsed.score === 0 && parsed.matched_keywords.length > 0) {
      const total =
        parsed.matched_keywords.length + parsed.missing_keywords.length;
      parsed.score =
        total > 0
          ? Math.round((parsed.matched_keywords.length / total) * 100)
          : 0;
      if (parsed.score >= 80) parsed.grade = "Excellent";
      else if (parsed.score >= 60) parsed.grade = "Good";
      else if (parsed.score >= 40) parsed.grade = "Fair";
      else parsed.grade = "Poor";
    }
    return parsed;
  } catch (_) {
    const fallbackPrompt = `${prompt}

IMPORTANT: The JSON "score" field must be a real calculated integer per the weighted formula (not 0 unless denominator is 0). "missing_keywords" must list every job keyword absent from the resume.

Example shape only (use REAL computed values, not these placeholders):
{
  "score": 72,
  "grade": "Good",
  "title_match": "close",
  "matched_keywords": ["Python", "REST APIs", "Docker", "PostgreSQL"],
  "missing_keywords": ["Kubernetes", "AWS", "CI/CD", "Terraform"],
  "must_have_matched": ["Python", "REST APIs", "PostgreSQL"],
  "must_have_missing": ["Kubernetes", "AWS"],
  "nice_to_have_matched": ["Docker"],
  "nice_to_have_missing": ["CI/CD", "Terraform"],
  "weak_matches": ["Docker"],
  "suggestions": "KEYWORDS: Add Kubernetes and AWS to your Experience or Projects section. PHRASING: Move Docker from your skills list into a project bullet to convert a weak match into a strong match. STRUCTURE: Include the exact job title in your summary line for a title-match bonus."
}`;

    const raw = await callGemini(fallbackPrompt, apiKey, {
      temperature: 0.1,
      maxOutputTokens: 2048,
      thinkingBudget: 0,
    });

    try {
      const parsed = normalizeATSResult(
        parseJsonSafely(
          raw,
          "Could not parse ATS score response. Please try again.",
        ),
      );
      if (parsed.score === 0 && parsed.matched_keywords.length > 0) {
        const total =
          parsed.matched_keywords.length + parsed.missing_keywords.length;
        parsed.score =
          total > 0
            ? Math.round((parsed.matched_keywords.length / total) * 100)
            : 0;
        if (parsed.score >= 80) parsed.grade = "Excellent";
        else if (parsed.score >= 60) parsed.grade = "Good";
        else if (parsed.score >= 40) parsed.grade = "Fair";
        else parsed.grade = "Poor";
      }
      return parsed;
    } catch (_) {
      const labeledPrompt = `${prompt}

Respond in exactly this plain-text format with real computed values (one field per line, comma-separated keyword lists):
Score: <integer 0-100>
Grade: <Excellent|Good|Fair|Poor>
Title Match: <exact|close|different|none>
Matched Keywords: keyword1, keyword2, keyword3
Missing Keywords: keyword4, keyword5, keyword6
Must Have Matched: keyword1, keyword2
Must Have Missing: keyword4, keyword5
Nice To Have Matched: keyword3
Nice To Have Missing: keyword6
Weak Matches: keyword2
Suggestions: KEYWORDS: <add these>. PHRASING: <one rewording>. STRUCTURE: <one structural fix>.`;

      const labeledRaw = await callGemini(labeledPrompt, apiKey, {
        temperature: 0.1,
        maxOutputTokens: 1024,
        thinkingBudget: 0,
      });

      return parseLabeledATSResponse(labeledRaw);
    }
  }
}

/**
 * Build the stable system instruction for the application chatbot.
 *
 * This is the cached prefix: it must be built from stable inputs only (resume,
 * name, the open job) and stay byte-identical across messages in a session so
 * Gemini's implicit caching keeps discounting the repeated resume tokens.
 */
function buildChatSystemInstruction({ resumeText, candidateName, job }) {
  const name = candidateName || "the candidate";

  let jobBlock = "";
  if (job && (job.title || job.company || job.description)) {
    jobBlock = `

THE ROLE BEING APPLIED TO (use this to tailor answers; if a question is generic, you may ignore it):
Title: ${job.title || "Not specified"}
Company: ${job.company || "Not specified"}
Description:
${job.description || "Not provided"}`;
  }

  return `You are helping ${name} answer questions on a job application form. You write the answer AS ${name}, in the first person ("I"), as if ${name} typed it personally.

${name}'s RESUME (this is the ONLY source of truth about ${name} — never invent anything beyond it):
${resumeText}${jobBlock}

HOW TO WRITE (this is the whole point — it must read like a real person, not AI):
- Sound like a real human typing a thoughtful application answer. Natural, warm, professional. Not robotic, not salesy.
- Vary your sentence length. Mix short, punchy sentences with longer ones. Use plain, everyday words.
- Be specific. Pull real details from the resume (actual projects, tools, roles, outcomes) instead of vague claims.
- Write in first person as ${name}. Never refer to "the candidate" and never talk about yourself as an AI.
- Match the length to the question. A short question gets 2-4 sentences. An open "tell us about..." question gets one or two short paragraphs. Never pad.

NEVER DO THIS (these are dead giveaways of AI writing):
- No "As an AI", "I'd be happy to", "Certainly!", "Great question", or any preamble. Output only the answer itself.
- No corporate clichés: "passionate", "results-driven", "detail-oriented", "team player", "self-starter", "go-getter", "synergy", "hard-working", "think outside the box".
- No AI-tell words: "leverage", "utilize", "delve", "tapestry", "testament", "seamless", "robust", "game-changer", "in today's fast-paced world", "navigate the landscape".
- No markdown, no bullet points, no headings, no emojis — unless the question explicitly asks for a list. Just plain sentences the user can paste directly.
- No exclamation points unless one genuinely fits.

TRUTHFULNESS:
- Only use skills, experiences, and facts that are in the resume. Never invent employers, job titles, dates, metrics, or technologies ${name} has not listed.
- If a question asks about something not in the resume, answer honestly and pivot to a genuine, relevant strength — do not fabricate experience.

Reply with ONLY the answer text, ready to paste into the application.`;
}

/**
 * Answer a single application question with full resume (+ optional job) context
 * and the recent conversation history, in a natural, human voice.
 */
async function generateApplicationAnswer(
  { question, history, resumeText, candidateName, job },
  apiKey,
) {
  const systemInstruction = buildChatSystemInstruction({
    resumeText,
    candidateName,
    job,
  });

  const contents = [];
  (history || []).forEach((m) => {
    const role = m.role === "model" ? "model" : "user";
    const text = String(m.text ?? "").trim();
    if (text) contents.push({ role, parts: [{ text }] });
  });
  contents.push({
    role: "user",
    parts: [{ text: String(question ?? "").trim() }],
  });

  return await callGeminiConversation({ systemInstruction, contents }, apiKey, {
    temperature: 0.8,
    maxOutputTokens: 1024,
  });
}
