"""
ResumeMatch AI - Automated 20-JD Benchmark Test Suite
Validates universal requirement extraction, two-tier classification, and CPU latency across 20 diverse real-world JDs.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server

BENCHMARK_JDS = [
    {
        "id": "01_valuelabs_fullstack",
        "title": "ValueLabs Full Stack Developer (Bengaluru)",
        "domain": "Full Stack / Banking",
        "text": """We’re Hiring at ValueLabs | Full Stack Developer – Bengaluru (WFO) 🌍
We are looking for experienced Full Stack Developers with expertise in Node.js, React.js, TypeScript, GenAI, Automation Testing (Playwright), and Banking Domain experience for an exciting opportunity in Bengaluru.
Position Details
Role: Full Stack Developer
Experience: 5+ and 7+ Years
Location: Bengaluru
Domain: Banking
Notice Period: Immediate to 30 Days Preferred
Required Skills
Node.js
React.js
TypeScript
GenAI
Automation Testing Framework – Playwright
Banking Domain Experience (Mandatory)
If you are interested or know someone who would be a great fit, please share your updated resume at: Jobs.bfs2@valuelabs.com"""
    },
    {
        "id": "02_aws_serverless",
        "title": "Full Stack Serverless Engineer",
        "domain": "Cloud / Serverless",
        "text": """🚀 WE’RE HIRING | FULL STACK DEVELOPER
We’re looking for an experienced Full Stack Developer to build scalable applications and reliable, event-driven systems.
Key Requirements:
✅️ 6+ years of experience in full-stack development.
✅️ Strong expertise in TypeScript, Node.js, React, and GraphQL.
✅️ Hands-on experience with AWS serverless services: Lambda, AppSync, DynamoDB, EventBridge, and SQS.
✅️ Experience with event-driven architecture, APIs, webhooks, and production systems.
✅️ Strong understanding of system design, reliability, and end-to-end feature ownership.
✅️ Familiarity with React Native and AI-powered development tools is a plus.
💼 Tech Stack: AWS | TypeScript | Node.js | React | GraphQL | DynamoDB"""
    },
    {
        "id": "03_fintech_frontend",
        "title": "Associate Frontend Engineer (React / Next.js)",
        "domain": "Frontend / FinTech",
        "text": """📢 WE’RE HIRING: ASSOCIATE FRONTEND ENGINEER (REACT / NEXT.JS)
A FinTech and Digital Identity Technology Product Company is looking for an Associate Frontend Engineer.
🔑 KEY SKILLS
▪️ React, Next.js & TypeScript
▪️ Node.js & REST APIs
▪️ State Management & Data Fetching
▪️ CSS & Responsive Design
▪️ Git
▪️ Cypress / Playwright
▪️ Data Visualization
▪️ AI Coding Tools
📌 MANDATORY REQUIREMENTS
▪️ Minimum 1+ years of hands-on experience with React and TypeScript in production applications.
▪️ Practical experience with Next.js (routing, rendering and API routes).
▪️ Product company experience is mandatory; FinTech experience is a plus."""
    },
    {
        "id": "04_kyara_mern_python",
        "title": "Development Team Lead / Head of Dev",
        "domain": "Engineering Leadership",
        "text": """🚀 Hiring: Development Team Lead / Head of Development
📍 Location: Bodakdev, Ahmedabad
💼 Experience: 5–7 Years
💰 CTC: ₹21–25 LPA
Must-Have Skills:
Strong hands-on experience in Python & MERN Stack
Expertise in Backend & Frontend Development
Proven experience as a Tech Lead / Team Lead
Strong People Management & Team Leadership skills
Knowledge of SDLC, Agile, Code Reviews & Software Delivery
Understanding of Cloud Technologies & Microservices Architecture"""
    },
    {
        "id": "05_python_fastapi",
        "title": "Senior Backend Engineer (Python / FastAPI)",
        "domain": "Backend / Python",
        "text": """Senior Backend Engineer - Python / FastAPI
What We Are Looking For:
- Strong hands-on experience with Python 3.10+, FastAPI, and asyncio
- Experience designing RESTful APIs and WebSocket architectures
- Deep knowledge of PostgreSQL, query optimization, and Alembic migrations
- Experience with Redis caching and Celery background workers
- Docker, CI/CD pipelines, and AWS deployment
- Experience with microservices architecture and unit testing (pytest)
Nice to Have:
- Familiarity with Kafka or RabbitMQ event streaming
- Experience with GraphQL APIs is a bonus"""
    },
    {
        "id": "06_java_spring_microservices",
        "title": "Senior Java Microservices Architect",
        "domain": "Enterprise Java",
        "text": """Enterprise Java Engineer - Microservices
Required Skills:
1. Java 17+, Spring Boot 3, and Spring Cloud
2. Microservices Architecture & Distributed Systems
3. Relational databases: Oracle or PostgreSQL
4. Kafka messaging and event-driven architecture
5. Docker, Kubernetes, and Helm charts
6. Jenkins, Git, and automated testing with JUnit and Mockito
Preferred Qualifications:
- AWS or GCP cloud certification is desirable
- Experience in high-volume transaction processing is a plus"""
    },
    {
        "id": "07_golang_backend",
        "title": "Go / Golang Infrastructure Engineer",
        "domain": "Systems / Golang",
        "text": """We are hiring a Backend Engineer proficient in Golang.
Key Responsibilities & Requirements:
Must Have:
- 3+ years writing concurrent Go services (goroutines, channels)
- Experience with gRPC, Protocol Buffers, and RESTful APIs
- Working knowledge of PostgreSQL, Redis, and Elasticsearch
- Linux systems programming, Docker, and Kubernetes
- Understanding of distributed tracing and Prometheus metrics
Optional:
- Experience with Rust or C++ is a plus"""
    },
    {
        "id": "08_devops_cloud_engineer",
        "title": "Senior Cloud DevOps Engineer",
        "domain": "DevOps / Infrastructure",
        "text": """Senior Cloud DevOps Engineer
Requirements:
• 5+ years supporting AWS cloud infrastructure (VPC, EC2, ECS, EKS, IAM)
• Infrastructure as Code: Terraform and Terragrunt
• Container orchestration using Kubernetes and Docker
• CI/CD pipeline automation with GitHub Actions or GitLab CI
• Observability: Prometheus, Grafana, ELK Stack, Datadog
• Scripting proficiency in Bash and Python
Good to Have:
• Multi-cloud experience with Azure or GCP is an added advantage
• Knowledge of ArgoCD and GitOps workflows"""
    },
    {
        "id": "09_data_engineer",
        "title": "Data Engineer (PySpark / Databricks)",
        "domain": "Data Engineering",
        "text": """HIRING: DATA ENGINEER
Qualifications:
- Solid background in Data Engineering and ETL pipeline development
- Strong programming in Python and Advanced SQL
- Hands-on experience with Apache Spark (PySpark) and Databricks
- Experience with cloud data warehouses: Snowflake, Redshift, or BigQuery
- Workflow orchestration with Apache Airflow
- Understanding of data modeling (Star Schema, Data Vault)
Bonus Points:
- Experience with dbt (data build tool) is a plus
- Streaming data with Apache Kafka is desirable"""
    },
    {
        "id": "10_genai_llm_engineer",
        "title": "Generative AI & LLM Engineer",
        "domain": "Artificial Intelligence",
        "text": """Generative AI Application Engineer
Core Technical Requirements:
- Python proficiency with deep understanding of LLM APIs (OpenAI, Anthropic, Gemini)
- Hands-on experience with RAG architectures and Vector Databases (Pinecone, Chroma, Qdrant)
- Experience building AI agents using LangChain or LlamaIndex
- Prompt engineering, evaluation frameworks, and fine-tuning techniques
- FastAPI backend development and Docker containerization
Preferred:
- Experience deploying local models with Ollama or vLLM
- Frontend experience in React or Next.js is a plus"""
    },
    {
        "id": "11_mobile_flutter_react_native",
        "title": "Cross-Platform Mobile Engineer",
        "domain": "Mobile Development",
        "text": """Mobile App Developer (Flutter & React Native)
Must-Have Skills:
- Hands-on experience building mobile apps in Flutter (Dart) or React Native (TypeScript)
- State management: Bloc/Riverpod for Flutter, Redux/Zustand for React Native
- Integrating RESTful APIs and offline-first SQLite/Hive caching
- Publishing apps to Apple App Store and Google Play Store
- Mobile CI/CD with Fastlane
Nice to Have:
- Native iOS (Swift) or Android (Kotlin) bridge development is a plus"""
    },
    {
        "id": "12_angular_frontend",
        "title": "Senior Angular Developer",
        "domain": "Frontend / Angular",
        "text": """Senior Frontend Developer - Angular
Requirements:
* 4+ years of professional development with Angular (v14+) and TypeScript
* RxJS reactive programming and NgRx state management
* HTML5, SCSS, responsive design, and CSS Grid/Flexbox
* Unit testing with Jasmine and Karma
* Git version control and REST API integration
Preferred:
* Experience with Cypress automated testing is a plus
* Angular Material design framework experience is desirable"""
    },
    {
        "id": "13_php_laravel_fullstack",
        "title": "Full Stack PHP / Laravel Developer",
        "domain": "Full Stack / PHP",
        "text": """Full Stack Laravel Developer
Job Requirements:
- Expertise in PHP 8+, Laravel framework, and Eloquent ORM
- Frontend skills in Vue.js or React.js with Tailwind CSS
- Relational database design: MySQL or MariaDB
- Caching with Redis and queue workers
- RESTful API development and authentication (Sanctum/Passport)
- Git, Composer, and Linux Apache/Nginx configuration"""
    },
    {
        "id": "14_qa_automation_playwright",
        "title": "Lead QA Automation Engineer",
        "domain": "Quality Assurance",
        "text": """Lead QA Automation Engineer
Key Requirements:
- 5+ years in automated testing for web applications
- Strong hands-on coding in TypeScript or JavaScript with Playwright
- Experience building End-to-End (E2E) testing frameworks from scratch
- API testing with Postman or REST-Assured
- Integrating test suites into CI/CD pipelines (GitHub Actions, Jenkins)
- Performance testing with k6 or JMeter
- Strong debugging and defect tracking skills"""
    },
    {
        "id": "15_healthcare_healthtech",
        "title": "Full Stack HealthTech Engineer",
        "domain": "Healthcare / Compliance",
        "text": """HealthTech Full Stack Engineer
What You Will Need:
- Full-stack web development with React and Node.js
- TypeScript, Express.js, and PostgreSQL
- HIPAA compliance and secure patient data handling
- Knowledge of FHIR or HL7 healthcare standards (Mandatory)
- Docker, AWS, and secure RESTful APIs
Bonus:
- Telehealth WebRTC experience is a plus"""
    },
    {
        "id": "16_ecommerce_mern",
        "title": "E-Commerce Platform Engineer",
        "domain": "E-Commerce",
        "text": """E-Commerce Developer
Skills Required:
- React.js, Next.js, and Node.js
- GraphQL API integration and Shopify Storefront API
- Payment gateway integration (Stripe, Razorpay, or PayPal)
- Redux Toolkit, Tailwind CSS, and Web Performance Optimization
- Experience with high-traffic e-commerce storefronts
Nice to Have:
- Headless CMS experience (Strapi, Sanity) is a bonus"""
    },
    {
        "id": "17_cybersecurity_cloud_security",
        "title": "Cloud Security & DevSecOps Engineer",
        "domain": "Cybersecurity",
        "text": """Cloud Security Engineer
Must Have:
- AWS Security Hub, IAM, GuardDuty, and CloudTrail
- Vulnerability scanning with Trivy, Snyk, or SonarQube
- SAST, DAST, and container security for Docker and Kubernetes
- Secrets management: HashiCorp Vault or AWS Secrets Manager
- CI/CD security gating and compliance monitoring
- Python or Bash scripting for automation"""
    },
    {
        "id": "18_pipe_separated_raw",
        "title": "Pipe-Separated LinkedIn Fast Post",
        "domain": "Raw Quick Post",
        "text": """🔥 IMMEDIATE OPENING | FULL STACK DEVELOPER
Tech Stack: AWS | TypeScript | Node.js | React | GraphQL | DynamoDB | Redis | Docker
Experience: 4-6 Years
Location: Remote / India
Interested candidates apply now!"""
    },
    {
        "id": "19_unbulleted_paragraph",
        "title": "Unbulleted Continuous Paragraph Style",
        "domain": "Continuous Text",
        "text": """We are looking for an experienced Full Stack Software Engineer to build scalable microservices. The candidate must have extensive hands-on experience developing with React.js, Node.js, and TypeScript. You should be comfortable designing RESTful APIs, configuring PostgreSQL databases, deploying Docker containers on AWS, and writing automated unit tests. Experience with Redis caching and Git version control is essential."""
    },
    {
        "id": "20_mixed_clauses_semicolons",
        "title": "Mixed Conjunctions & Semicolons",
        "domain": "Complex Grammar",
        "text": """Required Qualifications:
▪️ Proficiency in Python, Django, and PostgreSQL is mandatory; experience with React.js is a plus.
▪️ Strong understanding of Docker containerization and CI/CD pipelines.
▪️ AWS cloud experience is required, whereas Azure experience is optional.
▪️ Hands-on with Cypress or Playwright for automated testing."""
    }
]

def run_benchmark_suite():
    print("=" * 80)
    print("🚀 RESUMEMATCH AI - 20-JD AUTOMATED BENCHMARK SUITE")
    print("=" * 80)
    print(f"Total JDs to test: {len(BENCHMARK_JDS)}")
    
    start_all = time.time()
    passed = 0
    failed = 0
    results = []

    for idx, item in enumerate(BENCHMARK_JDS, 1):
        jd_id = item["id"]
        title = item["title"]
        text = item["text"]
        
        t0 = time.time()
        try:
            tiered = server.extract_universal_criteria_tiered(text)
            core = tiered["core"]
            preferred = tiered["preferred"]
            total_crit = len(tiered["all"])
            latency = (time.time() - t0) * 1000

            # Quality assertions
            errors = []
            if total_crit < 2:
                errors.append(f"Too few criteria extracted ({total_crit})")
            if total_crit > 30:
                errors.append(f"Excessive criteria extracted ({total_crit})")

            # Check for bad artifacts
            all_crit = core + preferred
            for c in all_crit:
                if '(' in c and ')' not in c:
                    errors.append(f"Unclosed paren in '{c}'")
                if ')' in c and '(' not in c:
                    errors.append(f"Unmatched paren in '{c}'")
                if any(bad in c.lower() for bad in ["(mandatory)", "(required)", "experience mandatory", "is a plus"]):
                    errors.append(f"Unstripped qualifier in '{c}'")

            if not errors:
                passed += 1
                status = "✅ PASS"
            else:
                failed += 1
                status = "❌ FAIL: " + "; ".join(errors[:2])

            results.append({
                "num": idx,
                "title": title[:36],
                "core": len(core),
                "pref": len(preferred),
                "latency_ms": round(latency, 1),
                "status": status,
                "sample_core": core[:3]
            })

        except Exception as e:
            failed += 1
            results.append({
                "num": idx,
                "title": title[:36],
                "core": 0,
                "pref": 0,
                "latency_ms": 0,
                "status": f"❌ EXCEPTION: {str(e)}",
                "sample_core": []
            })

    total_time = (time.time() - start_all) * 1000

    # Print Table
    print("\n" + "-" * 80)
    print(f"{'#':<3} | {'Job Description Title':<38} | {'Core':<4} | {'Pref':<4} | {'Latency':<8} | {'Status'}")
    print("-" * 80)
    for r in results:
        print(f"{r['num']:<3} | {r['title']:<38} | {r['core']:<4} | {r['pref']:<4} | {r['latency_ms']:>5.1f}ms  | {r['status']}")
    print("-" * 80)

    print(f"\n📊 BENCHMARK SUMMARY:")
    print(f"Total JDs Screened: {len(BENCHMARK_JDS)}")
    print(f"Passed: {passed} / {len(BENCHMARK_JDS)} ({passed / len(BENCHMARK_JDS) * 100:.1f}%)")
    print(f"Failed: {failed}")
    print(f"Total Benchmark Execution Time: {total_time:.1f} ms (Avg: {total_time/len(BENCHMARK_JDS):.1f} ms/JD on CPU)")
    print("=" * 80)

    return passed == len(BENCHMARK_JDS)

if __name__ == "__main__":
    success = run_benchmark_suite()
    sys.exit(0 if success else 1)

