# RAG Cognitive Agent: Intelligent Knowledge Assistant
## User Manual

---

**Application Name:** RAG Cognitive Agent (Retrieval Augmented Generation Cognitive Agent)  
**Version:** 1.0  
**Last Updated:** December 2025

---

## Table of Contents

1. [Introduction](#introduction)
2. [What is V-Agents?](#what-is-v-agents)
3. [Key Features](#key-features)
4. [Getting Started](#getting-started)
5. [Using the Knowledge Assistant](#using-the-knowledge-assistant)
6. [Using the SQL Agent](#using-the-sql-agent)
7. [Managing Your Knowledge Base](#managing-your-knowledge-base)
8. [Voice Conversations](#voice-conversations)
9. [Admin Features](#admin-features)
10. [Troubleshooting](#troubleshooting)
11. [Frequently Asked Questions](#frequently-asked-questions)

---

## Introduction

Welcome to RAG Cognitive Agent! This user manual will guide you through using the application to get answers from your documents and databases using simple, natural language.

### Who is this for?

This application is designed for:
- Business professionals who need quick answers from company documents
- Team members who want to query databases without knowing SQL
- Managers who need insights from data and documents
- Anyone who wants an intelligent assistant for their knowledge base

---

## What is RAG Cognitive Agent?

RAG Cognitive Agent is an intelligent assistant that helps you:

1. **Ask questions about your documents** - Upload PDFs, Word files, Excel sheets, and more, then ask questions in plain English
2. **Query your databases** - Get data from your company database by asking questions naturally, without writing SQL code
3. **Have voice conversations** - Talk to your assistant using your microphone
4. **Get instant answers** - Receive accurate responses based on your actual data and documents

### Technology Stack

RAG Cognitive Agent is built using modern, enterprise-grade technologies:

**Frontend (What you see):**
- React - Modern web interface
- JavaScript - Interactive features
- Responsive design - Works on desktop and tablet

**Backend (Behind the scenes):**
- Python with Flask - Intelligent processing
- Node.js - Fast data handling
- AI/ML Models - Smart question understanding

**Database & Storage:**
- MySQL - Secure data storage
- Qdrant - Fast document search
- Vector embeddings - Intelligent matching

**AI Capabilities:**
- Natural Language Processing - Understands your questions
- RAG (Retrieval Augmented Generation) - Finds relevant information
- Text-to-SQL - Converts questions to database queries
- Voice Recognition - Understands spoken questions

---

## Key Features

### 📚 Knowledge Base Assistant
- Upload and organize documents by category
- Ask questions in plain English
- Get answers with source references
- Support for multiple file types (PDF, Word, Excel, PowerPoint, images, audio)

### 🗄️ SQL Agent (Database Assistant)
- Connect to your company database
- Ask questions about your data naturally
- See the SQL query that was generated
- Get results in easy-to-read format
- Auto-suggest table names while typing

### 🎤 Voice Conversations
- Speak your questions instead of typing
- Listen to responses
- Natural conversation flow
- Multiple voice options

### 👥 Team Collaboration
- Share knowledge bases with colleagues
- Role-based access control
- Track conversation history

### 🎭 AI Personas
- Different assistant personalities for different needs
- Customizable response styles
- Professional, friendly, or technical tones

---

## Getting Started

### Step 1: Accessing the Application

1. Open your web browser (Chrome, Firefox, or Edge recommended)
2. Navigate to the application URL provided by your administrator
3. You will see the login or home page

### Step 2: First-Time Setup

**For Administrators:**
1. Set up your company database connection (see Admin Features section)
2. Create knowledge base categories
3. Invite team members

**For Users:**
1. Your administrator will provide you with access
2. You can start asking questions immediately
3. Upload documents to your personal categories if allowed

---

## Using the Knowledge Assistant

### Creating a Knowledge Base

**Step 1: Create a Category**
1. Click on the **"Dashboard"** tab
2. Click **"+ New Category"** button
3. Enter a name for your category (e.g., "HR Policies", "Product Manuals")
4. Click **"Create"**

**Step 2: Upload Documents**
1. Click on your newly created category
2. Click **"Upload Files"** button
3. Select one or more files from your computer
4. Supported formats:
   - Documents: PDF, Word (.docx), PowerPoint (.pptx), Text files
   - Spreadsheets: Excel (.xlsx), CSV
   - Media: Images (PNG, JPG), Audio files (MP3, WAV)
5. Wait for the upload to complete
6. Click **"Index Files"** to make them searchable

**Step 3: Wait for Indexing**
- The system processes your documents (this may take a few minutes)
- You'll see a green "ACTIVE" status when ready
- Now you can start asking questions!

### Asking Questions

**Step 1: Select Your Category**
1. Click on **"Query RAG"** tab
2. From the dropdown, select the category you want to ask about
3. The category must show "ACTIVE" status

**Step 2: Type Your Question**
1. In the text box at the bottom, type your question naturally
   - Example: "What is the vacation policy?"
   - Example: "How do I reset my password?"
   - Example: "What are the product specifications?"
2. Click the **Send** button (arrow icon) or press Enter

**Step 3: Review the Answer**
1. The assistant will search through your documents
2. You'll receive an answer based on the information found
3. Source documents are referenced in the response
4. You can ask follow-up questions for clarification

### Tips for Better Results

✅ **DO:**
- Ask specific questions
- Use complete sentences
- Ask one question at a time
- Provide context if needed

❌ **DON'T:**
- Ask questions about information not in your documents
- Use very vague or broad questions
- Expect the assistant to know information outside your knowledge base

---

## Using the SQL Agent

The SQL Agent helps you get data from your company database without knowing SQL code.

### Setting Up Database Connection (Admin Only)

**Step 1: Navigate to Admin Tools**
1. Click on **"Admin Tools"** tab
2. Scroll to **"SQL Agent Configuration"** section

**Step 2: Enter Database Details**
1. **Host:** Your database server address (e.g., localhost or IP address)
2. **Port:** Usually 3306 for MySQL
3. **Database Name:** The name of your database
4. **Username:** Database user with read access
5. **Password:** Database password
6. Click **"Connect Database"**

**Step 3: Sync Database Schema**
1. After successful connection, click **"Sync Schema"**
2. Wait for the system to learn about your database tables
3. You'll see a success message when complete

### Asking Database Questions

**Step 1: Select SQL Agent**
1. Click on **"Query RAG"** tab
2. Select **"SQL Agent"** from the category dropdown
3. You'll see the database icon is highlighted

**Step 2: Ask Your Question**
1. Type your question in natural language:
   - Example: "How many customers do we have?"
   - Example: "Show me the top 10 products by sales"
   - Example: "What is the total revenue for last month?"
2. Click Send or press Enter

**Step 3: Review Results**
1. The system generates the SQL query automatically
2. You'll see:
   - **Generated SQL:** The actual database query (for reference)
   - **Results:** Your data in an easy-to-read format
3. You can ask follow-up questions to refine your results

### Using Table Name Autocomplete

When typing in SQL Agent mode, the system helps you with table names:

**How it works:**
1. Start typing a table name
2. A dropdown appears showing matching tables
3. Use your mouse or keyboard to select:
   - **Mouse:** Click on the table name
   - **Keyboard:** Use Arrow keys to navigate, Enter to select
4. The table name is automatically inserted

**Quick Access:**
- Click the **"Show Tables"** button to see all available tables
- Look at the number badge on the database icon to see how many tables are available

### SQL Agent Tips

✅ **Good Questions:**
- "List all active users"
- "What is the average order value?"
- "Show customers from California"
- "Count total orders this year"

❌ **Avoid:**
- Very complex multi-step analysis (break into smaller questions)
- Questions requiring data not in your database
- Requests to modify or delete data (read-only access)

---

## Managing Your Knowledge Base

### Viewing Your Categories

**Dashboard Overview:**
1. Click **"Dashboard"** tab
2. You'll see all your categories listed
3. Each category shows:
   - Name
   - Number of files
   - Index status (ACTIVE/INACTIVE)
   - Last updated date

### Adding More Documents

**To an Existing Category:**
1. Click on the category name
2. Click **"Upload Files"**
3. Select new files
4. Click **"Re-index"** to include new files in searches

### Sharing Knowledge Bases

**Share with Team Members:**
1. Click on a category
2. Click **"Share"** button
3. Select team members from the list
4. Choose their access level:
   - **View Only:** Can ask questions only
   - **Edit:** Can upload and manage files
5. Click **"Share"**

### Deleting Files or Categories

**Delete a File:**
1. Open the category
2. Find the file in the list
3. Click the **trash icon** next to the file
4. Confirm deletion
5. Re-index the category

**Delete a Category:**
1. Go to Dashboard
2. Click the **trash icon** on the category
3. Confirm deletion (this removes all files in the category)

---

## Voice Conversations

### Starting a Voice Conversation

**Step 1: Enable Voice Mode**
1. In the Query view, click the **microphone icon** at the bottom
2. Allow browser access to your microphone when prompted

**Step 2: Speak Your Question**
1. Click and hold the microphone button
2. Speak your question clearly
3. Release the button when done
4. The system transcribes your speech to text

**Step 3: Listen to Response**
1. The assistant processes your question
2. You'll see the text response
3. The response is read aloud automatically
4. You can continue the conversation by speaking again

### Voice Settings

**Adjust Voice Options:**
1. Click the **settings icon** in voice mode
2. Choose:
   - **Voice Type:** Male/Female, different accents
   - **Speech Speed:** Slower or faster
   - **Language:** Select your preferred language
3. Click **"Save"**

### Voice Conversation Tips

✅ **For Best Results:**
- Speak clearly and at a normal pace
- Use a quiet environment
- Keep questions concise
- Wait for the response before asking the next question

---

## Admin Features

### User Management

**Adding Users:**
1. Go to **Admin Tools**
2. Click **"User Management"**
3. Click **"Add User"**
4. Enter user details:
   - Name
   - Email
   - Role (Admin/Business User/Basic User)
5. Click **"Create User"**

**User Roles:**
- **Admin:** Full access to all features and settings
- **Business User:** Can create categories, upload files, share knowledge bases
- **Basic User:** Can only query shared knowledge bases

### Managing AI Personas

**Create a Persona:**
1. Go to **Admin Tools**
2. Click **"Persona Management"**
3. Click **"Create Persona"**
4. Enter:
   - **Name:** E.g., "Technical Support Agent"
   - **Description:** What this persona is for
   - **Behavior:** How it should respond (formal, friendly, technical)
5. Click **"Generate"** to let AI create the persona
6. Review and edit if needed
7. Click **"Save"**

**Assign Persona to Category:**
1. Go to Dashboard
2. Click on a category
3. Click **"Settings"**
4. Select a persona from the dropdown
5. Click **"Save"**

### API Key Management

**Add LLM API Keys:**
1. Go to **Admin Tools**
2. Click **"API Key Manager"**
3. Click **"Add API Key"**
4. Select provider (Groq, Gemini, etc.)
5. Enter your API key
6. Click **"Save"**

**Note:** API keys are required for the AI features to work. Contact your administrator if you need help obtaining keys.

### Compliance & Rules

**Set Compliance Rules:**
1. Go to **Admin Tools**
2. Click **"Compliance Profiles"**
3. Click **"Create Profile"**
4. Enter rules (e.g., "Do not share customer personal information")
5. Assign to categories that need these rules

---

## Troubleshooting

### Common Issues and Solutions

#### "No answer found" or "I don't have that information"

**Possible Causes:**
- The information is not in your uploaded documents
- Documents haven't been indexed yet
- Question is too vague

**Solutions:**
1. Check if the relevant document is uploaded
2. Verify the category status is "ACTIVE"
3. Try rephrasing your question more specifically
4. Make sure you selected the correct category

#### Upload Failed

**Possible Causes:**
- File is too large
- Unsupported file format
- Network connection issue

**Solutions:**
1. Check file size (maximum 50MB per file)
2. Verify file format is supported
3. Try uploading again
4. Contact your administrator if problem persists

#### SQL Agent Not Working

**Possible Causes:**
- Database not connected
- Schema not synced
- Database credentials incorrect

**Solutions:**
1. Check database connection status in Admin Tools
2. Click "Sync Schema" again
3. Verify database credentials with your administrator
4. Ensure you have read permissions on the database

#### Voice Not Working

**Possible Causes:**
- Microphone not connected
- Browser permissions not granted
- Audio settings incorrect

**Solutions:**
1. Check microphone is connected and working
2. Allow microphone access in browser settings
3. Try refreshing the page
4. Check voice settings in the application

#### Slow Performance

**Possible Causes:**
- Large knowledge base
- Many concurrent users
- Network issues

**Solutions:**
1. Break large categories into smaller ones
2. Clear browser cache
3. Check your internet connection
4. Contact administrator if issue persists

---

## Frequently Asked Questions

### General Questions

**Q: What types of files can I upload?**  
A: You can upload PDFs, Word documents (.docx), PowerPoint (.pptx), Excel (.xlsx), CSV files, text files, images (PNG, JPG), and audio files (MP3, WAV).

**Q: How many files can I upload?**  
A: There's no strict limit on the number of files, but each file should be under 50MB. Very large knowledge bases may take longer to process.

**Q: Can I upload files in different languages?**  
A: Yes, the system supports multiple languages. However, best results are achieved when questions and documents are in the same language.

**Q: How long does indexing take?**  
A: Indexing time depends on the number and size of files. Typically:
- Small documents (1-10 pages): 1-2 minutes
- Medium documents (10-100 pages): 5-10 minutes
- Large knowledge bases (100+ documents): 30+ minutes

**Q: Is my data secure?**  
A: Yes, all data is stored securely. Access is controlled by user roles and permissions. Only authorized users can access specific knowledge bases.

### Knowledge Base Questions

**Q: Can I update a document after uploading?**  
A: Yes, delete the old version and upload the new one, then re-index the category.

**Q: Why am I getting answers from the wrong documents?**  
A: Make sure you've selected the correct category. Each category is searched independently.

**Q: Can I search across multiple categories at once?**  
A: Currently, you need to select one category at a time. You can create a combined category with documents from multiple sources.

**Q: What happens to my conversation history?**  
A: Conversation history is saved per category and can be cleared anytime using the "Clear History" button.

### SQL Agent Questions

**Q: Do I need to know SQL to use the SQL Agent?**  
A: No! That's the whole point. Just ask questions in plain English, and the system generates the SQL for you.

**Q: Can I modify data in the database?**  
A: No, the SQL Agent has read-only access for security. It can only retrieve data, not modify it.

**Q: What if the SQL Agent generates the wrong query?**  
A: You can see the generated SQL and rephrase your question for better results. The more specific your question, the better the query.

**Q: Can I save frequently used queries?**  
A: Currently, you need to re-ask questions each time. You can copy the generated SQL for future reference.

### Voice Questions

**Q: What languages are supported for voice?**  
A: The system supports multiple languages including English, Spanish, French, German, and more. Check the voice settings for available options.

**Q: Can I use voice on mobile devices?**  
A: The application is optimized for desktop use. Mobile support may be limited depending on your device and browser.

**Q: Why is the voice recognition inaccurate?**  
A: Ensure you're in a quiet environment, speak clearly, and use a good quality microphone for best results.

---

## Getting Help

### Support Resources

**In-App Help:**
- Hover over any feature to see tooltips
- Look for the "?" icon for contextual help

**Contact Support:**
- Email: support@yourcompany.com
- Help Desk: [Your help desk URL]
- Phone: [Your support number]

**Training:**
- Request a training session from your administrator
- Check for video tutorials in your company portal

### Providing Feedback

We value your feedback! To report issues or suggest improvements:

1. Click the **"Feedback"** button (if available)
2. Or email your administrator with:
   - What you were trying to do
   - What happened
   - Screenshots if applicable
   - Your user ID and category name

---

## Appendix

### Glossary of Terms

**RAG (Retrieval Augmented Generation):** Technology that finds relevant information from documents before generating an answer.

**Vector Store:** A special database that stores documents in a way that makes them searchable by meaning, not just keywords.

**Indexing:** The process of analyzing and organizing documents so they can be searched quickly.

**Persona:** An AI personality that determines how the assistant responds to questions.

**SQL:** Structured Query Language - the language used to query databases (you don't need to know this!).

**Schema:** The structure of a database, including table names and column names.

**Embedding:** A mathematical representation of text that captures its meaning.

**LLM:** Large Language Model - the AI that understands and generates human-like text.

### Keyboard Shortcuts

**General:**
- `Ctrl + Enter` - Send message
- `Esc` - Close dropdown/modal
- `Ctrl + K` - Focus search/input

**SQL Agent:**
- `Arrow Up/Down` - Navigate table suggestions
- `Enter` - Select highlighted table
- `Esc` - Close suggestions

**Voice Mode:**
- `Space` - Hold to record
- `Esc` - Exit voice mode

---

## Document Information

**Version:** 1.0  
**Last Updated:** December 31, 2025  
**Document Owner:** RAG Cognitive Agent Development Team  
**Next Review Date:** March 2026

---

**Thank you for using RAG Cognitive Agent!**  
We hope this intelligent assistant makes your work easier and more productive.

For the latest updates and features, check with your administrator or visit the application regularly.
