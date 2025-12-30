// backend.js
/* eslint-disable no-console */
const express = require('express');
const multer = require('multer');
const path = require('path');
const fsp = require('fs').promises;
const fs = require('fs');
const axios = require('axios');
const cors = require('cors');
const os = require('os');
const winston = require('winston');
const FormData = require('form-data');
const mysql = require('mysql2/promise');
require('dotenv').config();

// ============================================================================
// 1) LOGGER & DIRECTORIES
// ============================================================================
const logDir = path.join(__dirname, 'logs');
if (!fs.existsSync(logDir)) fs.mkdirSync(logDir, { recursive: true });

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(winston.format.timestamp(), winston.format.json()),
  transports: [
    new winston.transports.File({
      filename: path.join(logDir, 'backend.log'),
      maxsize: 5 * 1024 * 1024,
      maxFiles: 5,
    }),
    new winston.transports.Console({ format: winston.format.simple() }),
  ],
});

const CHAT_HISTORY_DIR = path.join(__dirname, 'chat_histories');
const DB_DIR = path.join(__dirname, 'db');
const USERS_DB_PATH = path.join(DB_DIR, 'users.json');
const PERMISSIONS_DB_PATH = path.join(DB_DIR, 'permissions.json');
const RULEBOOKS_DB_PATH = path.join(DB_DIR, 'rulebooks.json');
const COMPLIANCE_DB_PATH = path.join(DB_DIR, 'compliance.json');
const TEST_QUESTIONS_DB_PATH = path.join(DB_DIR, 'test_questions.json');
const SHARES_DB_PATH = path.join(DB_DIR, 'shares.json');
const SQL_CONNECTIONS_DB_PATH = path.join(DB_DIR, 'sql_connections.json');
const UPLOAD_FOLDER = path.join(__dirname, 'data', 'uploads');

fs.mkdirSync(CHAT_HISTORY_DIR, { recursive: true });
fs.mkdirSync(DB_DIR, { recursive: true });
fs.mkdirSync(UPLOAD_FOLDER, { recursive: true });

// ============================================================================
// 2) EXPRESS, CORS & DB
// ============================================================================
const app = express();

const corsOptions = {
  origin: '*', // Allow all IPs
  optionsSuccessStatus: 200,
};

app.use(cors(corsOptions));
app.use(express.json({ limit: '20mb' }));

const AI_SERVER_URL = process.env.AI_SERVER_URL;

// Environment Detection
const hostname = os.hostname().toUpperCase();
const isLocal = ["Test"].some(keyword => hostname.includes(keyword));
logger.info(`🖥️ Server Hostname: ${hostname}`);
logger.info(`🏠 Is Local Environment: ${isLocal}`);

app.get('/api/env-check', (req, res) => {
  res.json({ isLocal, hostname });
});

// Database pool for the main RAG application (nrkindex_trn)
const dbPool = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_DATABASE,
  port: process.env.DB_PORT || 3306,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
}).on('error', (err) => {
  logger.error('MySQL Pool Error', { error: err.message });
});

// Database pool for the PMO database (pmo_prod)
const pmoDbPool = mysql.createPool({
  host: '88.150.227.111',
  user: 'pmo_prod_111',
  password: 'obiK^&(677',
  database: 'pmo_prod',
  port: 3306,
  connectionLimit: 10,
  waitForConnections: true,
  queueLimit: 0,
  multipleStatements: true,
  charset: 'utf8mb4',
}).on('error', (err) => {
  logger.error('PMO DB Pool Error', { error: err.message });
});


// ============================================================================
// 3) FS HELPERS
// ============================================================================

const readJsonDb = (filePath) => async () => {
  try {
    if (!fs.existsSync(filePath)) {
      await fsp.writeFile(filePath, JSON.stringify(filePath.endsWith('users.json') ? [] : {}));
      return filePath.endsWith('users.json') ? [] : {};
    }
    const data = await fsp.readFile(filePath, 'utf8');
    return JSON.parse(data || (filePath.endsWith('users.json') ? '[]' : '{}'));
  } catch (e) {
    logger.error(`readJsonDb failed for ${filePath}`, { error: e.message });
    return filePath.endsWith('users.json') ? [] : {};
  }
};

const writeJsonDb = (filePath) => async (data) => {
  try {
    await fsp.writeFile(filePath, JSON.stringify(data, null, 2));
  } catch (e) {
    logger.error(`writeJsonDb failed for ${filePath}`, { error: e.message });
  }
};

const readUsers = readJsonDb(USERS_DB_PATH);
const writeUsers = writeJsonDb(USERS_DB_PATH);
const readPermissions = readJsonDb(PERMISSIONS_DB_PATH);
const writePermissions = writeJsonDb(PERMISSIONS_DB_PATH);
const readRulebooks = readJsonDb(RULEBOOKS_DB_PATH);
const writeRulebooks = writeJsonDb(RULEBOOKS_DB_PATH);
const readCompliance = readJsonDb(COMPLIANCE_DB_PATH);
const writeCompliance = writeJsonDb(COMPLIANCE_DB_PATH);
const readTestQuestions = readJsonDb(TEST_QUESTIONS_DB_PATH);
const writeTestQuestions = writeJsonDb(TEST_QUESTIONS_DB_PATH);
const readShares = readJsonDb(SHARES_DB_PATH);
const writeShares = writeJsonDb(SHARES_DB_PATH);
const readSqlConnections = readJsonDb(SQL_CONNECTIONS_DB_PATH);
const writeSqlConnections = writeJsonDb(SQL_CONNECTIONS_DB_PATH);


const clearAiServerCache = async () => {
  try {
    await axios.post(`${AI_SERVER_URL}/clear-cache`);
    logger.info('Cleared AI server retriever cache.');
  } catch (e) {
    logger.warn('Failed to clear AI server cache', { error: e.response?.data || e.message });
  }
};

// ============================================================================
// 4) GENERIC PROXY
// ============================================================================
const proxyToAiServer =
  (route, method = 'post', shouldClearCache = false) =>
    async (req, res) => {
      const finalRoute = Object.keys(req.params).reduce(
        (acc, key) => acc.replace(`:${key}`, encodeURIComponent(req.params[key])),
        route
      );
      const endpoint = `${AI_SERVER_URL}/${finalRoute}`;
      logger.info('Proxying request to AI server', { method: method.toUpperCase(), endpoint });

      try {
        const isStream = route.includes('tts') || route.includes('demo') || route.includes('greeting');
        const config = {
          method,
          url: endpoint,
          data: req.body,
          params: req.query,
          ...(isStream ? { responseType: 'stream' } : {}),
        };

        const resp = await axios(config);

        if (shouldClearCache) await clearAiServerCache();

        if (isStream) {
          res.setHeader('Content-Type', resp.headers['content-type'] || 'audio/wav');
          if (resp.headers['content-disposition']) {
            res.setHeader('Content-Disposition', resp.headers['content-disposition']);
          }
          if (resp.headers['content-length']) {
            res.setHeader('Content-Length', resp.headers['content-length']);
          }
          resp.data.pipe(res);
        } else {
          res.status(resp.status).json(resp.data);
        }
      } catch (e) {
        const status = e.response?.status || 500;
        const payload = e.response?.data || { error: 'Internal AI server error' };
        logger.error(`Proxy for '${route}' failed`, { status, error: payload });
        return res.status(status).json(payload);
      }
    };

// ============================================================================
// 5) GUARD: prevent concurrent jobs
// ============================================================================
const activeJobs = new Map();
const JOB_TIMEOUT_MS = 10 * 60 * 1000;

const keyFor = (u, c, action) => `${u}::${c}::${action}`;

const guardedHandler = (action) => (req, res, next) => {
  const { username, category } = req.body || {};
  if (!username || !category) {
    return res.status(400).json({ error: 'username and category are required' });
  }

  const key = keyFor(username, category, action);
  const existingJob = activeJobs.get(key);

  if (existingJob) {
    if (Date.now() - existingJob.timestamp > JOB_TIMEOUT_MS) {
      logger.warn('Stale job found and removed', { key });
      activeJobs.delete(key);
    } else {
      return res
        .status(409)
        .json({ error: `Another '${action}' is already running for ${username}/${category}. Try again shortly.` });
    }
  }

  activeJobs.set(key, { timestamp: Date.now() });
  res.once('finish', () => activeJobs.delete(key));
  next();
};

// ============================================================================
// 6) FILE UPLOAD
// ============================================================================
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const { username, category } = req.body || {};
    if (!username || !category) return cb(new Error('Username and category are required for upload.'));
    const dir = path.join(UPLOAD_FOLDER, username, category);
    try {
      await fsp.mkdir(dir, { recursive: true });
      cb(null, dir);
    } catch (err) {
      cb(err);
    }
  },
  filename: (_req, file, cb) => cb(null, Buffer.from(file.originalname, 'latin1').toString('utf8')),
});
const upload = multer({ storage });

app.post('/api/rag/upload', upload.array('files'), async (req, res) => {
  try {
    const { username, category } = req.body || {};
    if (!username || !category) return res.status(400).json({ error: 'username and category are required' });

    const permissions = await readPermissions();
    const categoryId = `${username}-${category}`;
    if (!permissions[categoryId]) {
      permissions[categoryId] = {
        owner: username,
        categoryName: category,
        business: false,
        basic: false,
        personaId: null,
        complianceProfileId: null
      };
      await writePermissions(permissions);
      logger.info('Created new default permissions for category', { categoryId });
    }

    logger.info('File upload successful', { user: username, category, files: req.files?.length || 0 });
    await clearAiServerCache();

    res.json({
      message: `Successfully uploaded ${req.files.length} files.`,
      files: (req.files || []).map((f) => f.originalname),
    });
  } catch (e) {
    logger.error('Upload failed', { error: e.message });
    res.status(500).json({ error: 'Upload failed' });
  }
});

// ============================================================================
// 7) VOICE STT / TTS / DEMO
// ============================================================================
const memoryUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 },
});

app.post('/api/voice/stt', memoryUpload.single('audio'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No audio file provided' });
  try {
    const form = new FormData();
    form.append('audio', req.file.buffer, {
      filename: req.file.originalname || 'audio.webm',
      contentType: req.file.mimetype || 'audio/webm',
    });
    // Append other form fields from the request
    if (req.body.provider) {
      form.append('provider', req.body.provider);
    }
    if (req.body.firm_id) {
      form.append('firm_id', req.body.firm_id);
    }
    const resp = await axios.post(`${AI_SERVER_URL}/voice/stt`, form, { headers: form.getHeaders() });
    res.json(resp.data);
  } catch (e) {
    logger.error('STT proxy failed', { error: e.response?.data || e.message });
    res.status(e.response?.status || 500).json(e.response?.data || { error: 'Speech-to-text failed' });
  }
});

app.post('/api/voice/tts', proxyToAiServer('voice/tts'));
app.post('/api/voice/demo', proxyToAiServer('voice/demo'));
app.post('/api/voice/greeting', proxyToAiServer('voice/greeting'));
app.get('/api/voice/list-voices', proxyToAiServer('voice/list-voices', 'get'));
app.get('/api/voice/list-google-voices', proxyToAiServer('voice/list-google-voices', 'get'));
app.get('/api/voice/list-elevenlabs-voices', proxyToAiServer('voice/list-elevenlabs-voices', 'get'));
app.get('/api/voice/list-deepgram-voices', proxyToAiServer('voice/list-deepgram-voices', 'get'));

// ============================================================================


// ============================================================================
// 8) RAG MGMT & DATA
// ============================================================================
app.post('/api/rag/create-index', guardedHandler('create-index'), proxyToAiServer('create-index', 'post', true));
app.post('/api/rag/update-index', guardedHandler('update-index'), proxyToAiServer('update-index', 'post', true));
app.post('/api/rag/delete-index', guardedHandler('delete-index'), proxyToAiServer('delete-index', 'post', true));
app.post('/api/rag/delete-category', guardedHandler('delete-category'), proxyToAiServer('delete-category', 'post', true));

// ===================================
// FILE MANAGEMENT ENDPOINTS
// ===================================

const FILE_DESCRIPTIONS_DB_PATH = path.join(DB_DIR, 'file_descriptions.json');
const readFileDescriptions = readJsonDb(FILE_DESCRIPTIONS_DB_PATH);
const writeFileDescriptions = writeJsonDb(FILE_DESCRIPTIONS_DB_PATH);
const archiver = require('archiver');

// Get file descriptions for a user's category
app.get('/api/rag/files/descriptions', async (req, res) => {
  const { username, category } = req.query;
  if (!username || !category) return res.status(400).json({ error: 'Username and category are required' });

  try {
    const allDescriptions = await readFileDescriptions();
    const key = `${username}-${category}`;
    res.json(allDescriptions[key] || {});
  } catch (e) {
    logger.error('Failed to get file descriptions', { error: e.message });
    res.status(500).json({ error: 'Failed' });
  }
});

// Update a file description
app.post('/api/rag/files/description', async (req, res) => {
  const { username, category, filename, description } = req.body;
  if (!username || !category || !filename) return res.status(400).json({ error: 'Missing fields' });

  try {
    const allDescriptions = await readFileDescriptions();
    const key = `${username}-${category}`;
    if (!allDescriptions[key]) allDescriptions[key] = {};

    allDescriptions[key][filename] = description;
    await writeFileDescriptions(allDescriptions);
    res.json({ success: true });
  } catch (e) {
    logger.error('Failed to save file description', { error: e.message });
    res.status(500).json({ error: 'Failed' });
  }
});

// Download a single file
app.get('/api/rag/files/download', async (req, res) => {
  const { username, category, filename } = req.query;
  if (!username || !category || !filename) return res.status(400).send('Missing params');

  // Basic traversal protection
  if (filename.includes('..') || category.includes('..')) return res.status(403).send('Invalid path');

  const filePath = path.join(UPLOAD_FOLDER, username, category, filename);
  if (fs.existsSync(filePath)) {
    res.download(filePath);
  } else {
    res.status(404).send('File not found');
  }
});

// Download all files as ZIP
app.get('/api/rag/files/download-zip', async (req, res) => {
  const { username, category } = req.query;
  if (!username || !category) return res.status(400).send('Missing params');
  if (category.includes('..')) return res.status(403).send('Invalid path');

  const sourceDir = path.join(UPLOAD_FOLDER, username, category);
  if (!fs.existsSync(sourceDir)) return res.status(404).send('Category folder not found');

  res.attachment(`${category}.zip`);
  const archive = archiver('zip', { zlib: { level: 9 } });

  archive.on('error', (err) => {
    logger.error('Zip archive error', { error: err.message });
    res.status(500).send({ error: err.message });
  });

  archive.pipe(res);
  archive.directory(sourceDir, false);
  await archive.finalize();
});

// Delete a single file
app.delete('/api/rag/files/delete', async (req, res) => {
  const username = req.query.username || req.body.username;
  const category = req.query.category || req.body.category;
  const filename = req.query.filename || req.body.filename;

  if (!username || !category || !filename) return res.status(400).json({ error: 'Missing params' });
  if (filename.includes('..') || category.includes('..')) return res.status(403).json({ error: 'Invalid path' });

  const filePath = path.join(UPLOAD_FOLDER, username, category, filename);

  try {
    if (fs.existsSync(filePath)) {
      await fsp.unlink(filePath);

      // Also remove description if exists
      const allDescriptions = await readFileDescriptions();
      const key = `${username}-${category}`;
      if (allDescriptions[key] && allDescriptions[key][filename]) {
        delete allDescriptions[key][filename];
        await writeFileDescriptions(allDescriptions);
      }

      res.json({ message: 'File deleted' });
    } else {
      res.status(404).json({ error: 'File not found' });
    }
  } catch (e) {
    logger.error('Failed to delete file', { error: e.message });
    res.status(500).json({ error: 'Delete failed' });
  }
});

app.get('/api/rag/structure', async (req, res) => {
  try {
    const { username } = req.query;
    if (!username) return res.status(400).json({ error: 'Username query parameter is required' });

    const resp = await axios.get(`${AI_SERVER_URL}/structure/${encodeURIComponent(username)}`);
    const structureData = resp.data;
    const allPermissions = await readPermissions();
    let permissionsModified = false;

    if (structureData && structureData[username]) {
      structureData[username] = await Promise.all(structureData[username].map(async category => {
        const categoryId = `${username}-${category.name}`;
        let categoryPermissions = allPermissions[categoryId];

        if (!categoryPermissions) {
          permissionsModified = true;
          allPermissions[categoryId] = {
            owner: username,
            categoryName: category.name,
            business: false,
            basic: false,
            personaId: null,
            complianceProfileId: null
          };
          categoryPermissions = allPermissions[categoryId];
          logger.info('Auto-created missing permission entry for category', { categoryId });
        }

        // Enrich file list with stats from the filesystem
        let enrichedFiles = [];
        if (Array.isArray(category.files)) {
          enrichedFiles = await Promise.all(category.files.map(async (f) => {
            const fileName = typeof f === 'string' ? f : f.name;
            const filePath = path.join(UPLOAD_FOLDER, username, category.name, fileName);
            try {
              const stats = await fsp.stat(filePath);
              return { name: fileName, size: stats.size, added: stats.birthtime };
            } catch (ignore) {
              // If file not found on disk, return default
              return { name: fileName, size: 0, added: null };
            }
          }));
        }

        return {
          ...category,
          files: enrichedFiles,
          permissions: {
            business: categoryPermissions.business,
            basic: categoryPermissions.basic
          },
          personaId: categoryPermissions.personaId || null,
          complianceProfileId: categoryPermissions.complianceProfileId || null
        };
      }));

      if (permissionsModified) {
        await writePermissions(allPermissions);
      }
    }

    res.json(structureData);
  } catch (e) {
    logger.error('Structure endpoint failed', { error: e.response?.data || e.message });
    res.status(e.response?.status || 500).json({ error: 'Failed to fetch data structure' });
  }
});

// ============================================================================
// 9) CHAT HISTORY
// ============================================================================
const getHistoryFilePath = async (userId, userRole, category) => {
  const userHistoryDir = path.join(CHAT_HISTORY_DIR, encodeURIComponent(userRole), encodeURIComponent(userId));
  await fsp.mkdir(userHistoryDir, { recursive: true });
  const safeFilename = `${encodeURIComponent(category)}.json`;
  return path.join(userHistoryDir, safeFilename);
};

app.get('/api/chat/history/:username/:role/:category', async (req, res) => {
  const { username, role, category } = req.params;
  try {
    const filePath = await getHistoryFilePath(username, role, category);
    if (fs.existsSync(filePath)) {
      const data = await fsp.readFile(filePath, 'utf8');
      res.json(JSON.parse(data));
    } else {
      res.json([]);
    }
  } catch (e) {
    logger.error('Failed to read chat history', { username, role, category, error: e.message });
    res.status(500).json({ error: 'Failed to retrieve chat history.' });
  }
});

app.post('/api/chat/history/:username/:role/:category', async (req, res) => {
  const { username, role, category } = req.params;
  const chatHistory = req.body;
  if (!Array.isArray(chatHistory)) {
    return res.status(400).json({ error: 'Request body must be a chat history array.' });
  }
  try {
    const filePath = await getHistoryFilePath(username, role, category);
    await fsp.writeFile(filePath, JSON.stringify(chatHistory, null, 2));
    res.status(200).json({ message: 'History saved successfully.' });
  } catch (e) {
    logger.error('Failed to write chat history', { username, role, category, error: e.message });
    res.status(500).json({ error: 'Failed to save chat history.' });
  }
});

app.delete('/api/chat/history/:username/:role/:category', async (req, res) => {
  const { username, role, category } = req.params;
  try {
    const filePath = await getHistoryFilePath(username, role, category);
    if (fs.existsSync(filePath)) {
      await fsp.unlink(filePath);
      res.json({ message: 'History deleted.' });
    } else {
      res.status(404).json({ error: 'No history found.' });
    }
  } catch (e) {
    logger.error('Failed to delete chat history', { username, role, category, error: e.message });
    res.status(500).json({ error: 'Failed to delete history.' });
  }
});

// ============================================================================
// 10) CORE RAG QUERY
// ============================================================================
app.post('/api/rag/query', async (req, res) => {
  const { owner_id, category, question, session_id, queried_by_id, queried_by_role, persona_id, firmId } = req.body || {};

  const required = { owner_id, category, question, queried_by_id, queried_by_role, persona_id, firmId };
  const missing = Object.keys(required).filter(key => !required[key]);

  if (missing.length > 0) {
    return res.status(400).json({ error: `Missing required fields: ${missing.join(', ')}` });
  }

  const finalSessionId = session_id || `${queried_by_id}-${category}-${persona_id}`;

  const permissions = await readPermissions();
  const categoryId = `${owner_id}-${category}`;
  const categorySettings = permissions[categoryId] || {};
  const complianceProfileId = categorySettings.complianceProfileId || null;

  let complianceRules = null;
  if (complianceProfileId) {
    const complianceProfiles = await readCompliance();
    const profile = complianceProfiles[complianceProfileId];
    if (profile) {
      complianceRules = profile.content;
    }
  }

  logger.info('RAG query received', { owner: owner_id, querier: queried_by_id, category, sessionId: finalSessionId, persona_id, hasCompliance: !!complianceRules, firmId });

  try {
    const rulebooks = await readRulebooks();
    const rulebookKey = `${owner_id}-${category}`;
    const rulebookContent = rulebooks[rulebookKey] || '';

    const resp = await axios.post(`${AI_SERVER_URL}/rag/chain`, {
      ...req.body,
      session_id: finalSessionId,
      rulebook: rulebookContent,
      compliance_rules: complianceRules,
      firm_id: firmId,
    });
    res.json(resp.data);
  } catch (e) {
    logger.error('RAG query proxy failed', { error: e.response?.data || e.message });
    res.status(e.response?.status || 500).json(e.response?.data || { error: 'Failed to process query' });
  }
});


// ============================================================================
// 11) USERS & EMPLOYEES
// ============================================================================
app.post('/api/users/sync', async (req, res) => {
  const { id, name, role } = req.body || {};
  if (!id || !name || !role) return res.status(400).json({ error: 'id, name, role required' });

  const users = await readUsers();
  const found = users.find((u) => u.id === id);
  if (!found) {
    const newUser = { id, name, role };
    users.push(newUser);
    await writeUsers(users);
    logger.info('New user synced', { id, name, role });
    return res.status(201).json(newUser);
  }
  res.json(found);
});

app.get('/api/employees', async (req, res) => {
  // SECURITY FIX: Exclude the current user from the list of employees to share with.
  // NOTE: The frontend must be updated to send the current user's ID as a query parameter.
  // e.g., GET /api/employees?excludeId=123
  const { excludeId } = req.query;

  try {
    let sql = `
            SELECT EMPID, EMPNAME, STATUS, FIRM_ID, TYPE
            FROM EMPLOY_REGISTRATION
            WHERE STATUS = 'Active' AND TYPE = 'Employe'
        `;
    const params = [];

    if (excludeId) {
      sql += ' AND EMPID != ?';
      params.push(excludeId);
    }

    const [rows] = await pmoDbPool.query(sql, params);
    res.json(rows);
  } catch (e) {
    logger.error('Failed to fetch employees', { error: e.message });
    res.status(500).json({ error: 'Failed to fetch employee list.' });
  }
});


// ============================================================================
// 12) PERSONA MGMT
// ============================================================================
app.get('/api/personas', proxyToAiServer('personas', 'get'));
app.post('/api/personas', proxyToAiServer('personas', 'post'));
app.put('/api/personas/:persona_id', proxyToAiServer('personas/:persona_id', 'put'));
app.delete('/api/personas/:persona_id', proxyToAiServer('personas/:persona_id', 'delete'));

// ============================================================================
// 12.4) REAL-TIME FILE PROCESSING
// ============================================================================
app.post('/api/chat/process-file', memoryUpload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file provided' });
    }

    const { firm_id = 1 } = req.body;

    // Create form data for AI server
    const formData = new FormData();
    formData.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    });
    formData.append('firm_id', firm_id);

    // Forward to AI server
    const response = await axios.post(`${AI_SERVER_URL}/chat/process-file`, formData, {
      headers: {
        ...formData.getHeaders(),
      },
      timeout: 30000, // 30 second timeout for file processing
    });

    res.json(response.data);
  } catch (error) {
    logger.error('Chat file processing failed:', error.message);
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: 'File processing failed' });
    }
  }
});

// ============================================================================
// ============================================================================
// 12.5) SQL AGENT ENDPOINTS (New)
// ============================================================================
app.post('/api/sql_agent/connect', async (req, res) => {
  const { firm_id, db_config } = req.body;

  // 1. First, try to connect via AI Server (Validation)
  const proxy = proxyToAiServer('api/sql_agent/connect', 'post');

  // We wrap the proxy response to intercept success
  const originalJson = res.json;
  const originalStatus = res.status;

  let proxyStatusCode = 200;

  res.status = (code) => {
    proxyStatusCode = code;
    return originalStatus.call(res, code);
  };

  res.json = async (data) => {
    // If connection was successful, save to local JSON
    if (proxyStatusCode >= 200 && proxyStatusCode < 300) {
      try {
        const connections = await readSqlConnections();
        let firmConfigs = connections[firm_id];

        if (!firmConfigs) firmConfigs = [];
        if (!Array.isArray(firmConfigs)) firmConfigs = [firmConfigs]; // Migration

        const idx = firmConfigs.findIndex(c => c.database === db_config.database && c.host === db_config.host);
        if (idx !== -1) {
          firmConfigs[idx] = db_config;
        } else {
          firmConfigs.push(db_config);
        }

        connections[firm_id] = firmConfigs;
        await writeSqlConnections(connections);
        logger.info('Saved SQL connection locally for firm', { firm_id });
      } catch (e) {
        logger.error('Failed to save SQL connection locally', { error: e.message });
      }
    }
    return originalJson.call(res, data);
  };

  await proxy(req, res);
});

app.delete('/api/sql_agent/connect', async (req, res) => {
  const { firm_id, database } = req.body;

  const proxy = proxyToAiServer('api/sql_agent/connect', 'delete');

  const originalJson = res.json;
  const originalStatus = res.status;
  let proxyStatusCode = 200;

  res.status = (code) => {
    proxyStatusCode = code;
    return originalStatus.call(res, code);
  };

  res.json = async (data) => {
    if (proxyStatusCode >= 200 && proxyStatusCode < 300) {
      try {
        const connections = await readSqlConnections();
        let firmConfigs = connections[firm_id];

        if (firmConfigs) {
          if (database && Array.isArray(firmConfigs)) {
            connections[firm_id] = firmConfigs.filter(c => c.database !== database);
          } else {
            if (!database) delete connections[firm_id];
          }
          await writeSqlConnections(connections);
          logger.info('Deleted SQL connection locally for firm', { firm_id });
        }
      } catch (e) {
        logger.error('Failed to delete SQL connection locally', { error: e.message });
      }
    }
    return originalJson.call(res, data);
  };

  await proxy(req, res);
});

app.get('/api/sql_agent/config', async (req, res) => {
  const { firm_id } = req.query;
  try {
    const connections = await readSqlConnections();
    const config = connections[firm_id] || [];
    res.json(config);
  } catch (e) {
    logger.error('Failed to get SQL config', { error: e.message });
    res.status(500).json({ error: 'Failed to retrieve configuration' });
  }
});
app.get('/api/sql_agent/test', proxyToAiServer('api/sql_agent/test', 'get'));
// Redirect generate to the new RAG-enabled endpoint
app.post('/api/sql_agent/generate', proxyToAiServer('api/sql_agent_rag/generate', 'post'));
app.post('/api/sql_agent/sync', proxyToAiServer('api/sql_agent_rag/sync', 'post'));
app.get('/api/sql_agent/sync_status', proxyToAiServer('api/sql_agent_rag/status', 'get'));
app.post('/api/sql_agent/execute', proxyToAiServer('api/sql_agent/execute', 'post'));
app.post('/api/sql_agent/ask', proxyToAiServer('api/sql_agent/ask', 'post'));
app.get('/api/sql_agent/tables', proxyToAiServer('api/sql_agent/tables', 'get'));


// ============================================================================
// 13) COMPLIANCE MGMT
// ============================================================================
app.get('/api/compliance', async (req, res) => {
  try {
    const profiles = await readCompliance();
    res.json(Object.values(profiles));
  } catch (e) {
    logger.error('Failed to get compliance profiles', { error: e.message });
    res.status(500).json({ error: 'Failed to get compliance profiles' });
  }
});

app.post('/api/compliance', async (req, res) => {
  const { id, name, content } = req.body;
  if (!id || !name || typeof content === 'undefined') {
    return res.status(400).json({ error: 'id, name, and content are required.' });
  }
  try {
    const profiles = await readCompliance();
    profiles[id] = { id, name, content };
    await writeCompliance(profiles);
    logger.info('Compliance profile created/updated', { id });
    res.status(201).json(profiles[id]);
  } catch (e) {
    logger.error('Failed to save compliance profile', { error: e.message });
    res.status(500).json({ error: 'Failed to save compliance profile' });
  }
});

app.delete('/api/compliance/:profile_id', async (req, res) => {
  const { profile_id } = req.params;
  try {
    const profiles = await readCompliance();
    if (profile_id in profiles) {
      delete profiles[profile_id];
      await writeCompliance(profiles);
      logger.info('Compliance profile deleted', { id: profile_id });
    }
    res.status(204).send();
  } catch (e) {
    logger.error('Failed to delete compliance profile', { error: e.message });
    res.status(500).json({ error: 'Failed to delete compliance profile' });
  }
});

// ============================================================================
// 14) PERMISSIONS, RULEBOOKS & SETTINGS
// ============================================================================
app.put('/api/permissions/category', async (req, res) => {
  const { adminId, category, roleToUpdate, hasAccess } = req.body;
  if (!adminId || !category || !roleToUpdate || typeof hasAccess !== 'boolean') {
    return res.status(400).json({ error: 'Missing required fields' });
  }

  try {
    const categoryId = `${adminId}-${category}`;
    const permissions = await readPermissions();

    if (!permissions[categoryId]) {
      permissions[categoryId] = {
        owner: adminId,
        categoryName: category,
        business: false,
        basic: false
      };
    }
    permissions[categoryId][roleToUpdate] = hasAccess;

    await writePermissions(permissions);
    logger.info('Permissions updated', { admin: adminId, category, role: roleToUpdate, access: hasAccess });
    res.status(200).json({ message: 'Permissions updated successfully' });
  } catch (e) {
    logger.error('Failed to update permissions', { error: e.message });
    res.status(500).json({ error: 'Failed to update permissions on the server.' });
  }
});

app.put('/api/category/settings', async (req, res) => {
  const { adminId, categoryName, settings } = req.body;
  if (!adminId || !categoryName || !settings) {
    return res.status(400).json({ error: 'Missing required fields' });
  }

  try {
    const categoryId = `${adminId}-${categoryName}`;
    const permissions = await readPermissions();

    if (!permissions[categoryId]) {
      return res.status(404).json({ error: 'Category not found.' });
    }

    permissions[categoryId] = { ...permissions[categoryId], ...settings };

    await writePermissions(permissions);
    logger.info('Category settings updated', { admin: adminId, category: categoryName, settings });
    res.status(200).json({ message: 'Settings updated successfully' });
  } catch (e) {
    logger.error('Failed to update category settings', { error: e.message });
    res.status(500).json({ error: 'Failed to update settings on the server.' });
  }
});

app.get('/api/rag/rulebook/:adminId/:categoryName', async (req, res) => {
  const { adminId, categoryName } = req.params;
  try {
    const rulebooks = await readRulebooks();
    const rulebookKey = `${adminId}-${categoryName}`;
    const content = rulebooks[rulebookKey] || '';
    res.status(200).json({ content });
  } catch (e) {
    logger.error('Failed to get rulebook', { error: e.message });
    res.status(500).json({ error: 'Failed to retrieve rulebook.' });
  }
});

app.post('/api/rag/rulebook', async (req, res) => {
  const { adminId, category, rulebookContent } = req.body;
  if (!adminId || !category || typeof rulebookContent !== 'string') {
    return res.status(400).json({ error: 'adminId, category, and rulebookContent are required.' });
  }

  try {
    const rulebooks = await readRulebooks();
    const rulebookKey = `${adminId}-${category}`;
    rulebooks[rulebookKey] = rulebookContent;
    await writeRulebooks(rulebooks);
    logger.info('Rulebook saved', { key: rulebookKey });
    res.status(200).json({ message: 'Rulebook saved successfully.' });
  } catch (e) {
    logger.error('Failed to save rulebook', { error: e.message });
    res.status(500).json({ error: 'Failed to save rulebook.' });
  }
});

app.get('/api/rag/test-questions/:adminId/:categoryName', async (req, res) => {
  const { adminId, categoryName } = req.params;
  try {
    const allQuestions = await readTestQuestions();
    const key = `${adminId}-${categoryName}`;
    const questions = allQuestions[key] || '';
    res.status(200).json({ questions });
  } catch (e) {
    logger.error('Failed to get test questions', { error: e.message });
    res.status(500).json({ error: 'Failed to retrieve test questions.' });
  }
});

app.post('/api/rag/test-questions', async (req, res) => {
  const { adminId, category, questions } = req.body;
  if (!adminId || !category || typeof questions !== 'string') {
    return res.status(400).json({ error: 'adminId, category, and questions string are required.' });
  }
  try {
    const allQuestions = await readTestQuestions();
    const key = `${adminId}-${category}`;
    allQuestions[key] = questions;
    await writeTestQuestions(allQuestions);
    logger.info('Test questions saved', { key });
    res.status(200).json({ message: 'Test questions saved successfully.' });
  } catch (e) {
    logger.error('Failed to save test questions', { error: e.message });
    res.status(500).json({ error: 'Failed to save test questions.' });
  }
});

app.post('/api/rag/run-test', async (req, res) => {
  const { adminId, category, personaId, complianceProfileId, num_questions } = req.body;
  if (!adminId || !category) {
    return res.status(400).json({ error: 'adminId and category are required.' });
  }
  try {
    let complianceRules = null;
    if (complianceProfileId) {
      const complianceProfiles = await readCompliance();
      const profile = complianceProfiles[complianceProfileId];
      if (profile) {
        complianceRules = profile.content;
      }
    }

    const aiServerPayload = {
      owner_id: adminId,
      category: category,
      persona_id: personaId,
      compliance_rules: complianceRules,
      num_questions: num_questions || 10,
      firmId: req.body.firmId // Pass firmId for tests
    };

    const endpoint = `${AI_SERVER_URL}/rag/run-test`;
    logger.info('Proxying automated test run to AI server', { endpoint });

    const resp = await axios.post(endpoint, aiServerPayload, { timeout: 300000 }); // 5 minute timeout for tests
    res.status(resp.status).json(resp.data);

  } catch (e) {
    logger.error('RAG test run proxy failed', { error: e.response?.data || e.message });
    res.status(e.response?.status || 500).json(e.response?.data || { error: 'Failed to run test' });
  }
});


// UNIFIED ENDPOINT: Securely provides the correct list of RAGs for any user.
app.get('/api/rag/viewable', async (req, res) => {
  const { userId, userRole, firmId } = req.query;
  if (!userId || !userRole) {
    return res.status(400).json({ error: 'userId and userRole are required' });
  }

  try {
    const allPermissions = await readPermissions();
    let categoriesToStatusCheck = new Map();

    // 1. Fetch Firm-Level Categories (Primary)
    if (firmId) {
      try {
        const firmResp = await axios.get(`${AI_SERVER_URL}/structure/${encodeURIComponent(firmId)}`);
        const firmCats = firmResp.data?.[firmId] || [];

        firmCats.forEach(cat => {
          const key = `${firmId}-${cat.name}`;
          // Admin sees all. Others check permissions.
          if (userRole === 'admin') {
            categoriesToStatusCheck.set(key, { name: cat.name, owner: String(firmId) });
          } else {
            const perm = allPermissions[key];
            if (perm && (perm[userRole] === true)) {
              categoriesToStatusCheck.set(key, { name: cat.name, owner: String(firmId) });
            }
          }
        });
      } catch (e) {
        if (e.response?.status !== 404) logger.warn('Could not get firm categories', { firmId, error: e.message });
      }
    }

    // 2. Legacy/User Level (Keep for backward compatibility or mixed usage)
    // If user is admin, they can see their own legacy KBs
    if (userRole === 'admin') {
      try {
        const ownResp = await axios.get(`${AI_SERVER_URL}/structure/${encodeURIComponent(userId)}`);
        const userCats = ownResp.data?.[userId] || [];
        userCats.forEach(cat => categoriesToStatusCheck.set(`${userId}-${cat.name}`, { name: cat.name, owner: userId }));
      } catch (e) { /* ignore 404 */ }
    }

    // 3. Shared Categories
    const shares = await readShares();
    const userShares = shares[userId] || [];
    userShares.forEach(share => {
      if (share.ownerId && share.categoryName) {
        categoriesToStatusCheck.set(`${share.ownerId}-${share.categoryName}`, { name: share.categoryName, owner: String(share.ownerId) });
      }
    });

    const categoryList = Array.from(categoriesToStatusCheck.values());
    if (categoryList.length === 0) {
      return res.status(200).json([]);
    }

    const statusResp = await axios.post(`${AI_SERVER_URL}/batch-status-check`, { categories: categoryList });
    const activeRagsWithStatus = (statusResp.data || []).filter(c => c.indexStatus === 'ACTIVE');

    const finalActiveRags = activeRagsWithStatus.map(rag => {
      const categoryId = `${rag.owner}-${rag.name}`;
      const permissions = allPermissions[categoryId];
      return {
        ...rag,
        personaId: permissions ? (permissions.personaId || null) : null,
        complianceProfileId: permissions ? (permissions.complianceProfileId || null) : null
      };
    });

    res.status(200).json(finalActiveRags);

  } catch (e) {
    logger.error('Failed to get viewable categories', { userId, userRole, error: e.message });
    res.status(500).json({ error: 'Failed to retrieve categories.' });
  }
});

// ============================================================================
// 15) API KEY MANAGER ENDPOINTS
// ============================================================================
const parseEnum = (colType) => {
  const match = colType.match(/^enum\((.*)\)$/);
  if (!match) return [];
  return match[1].split(',').map(item => item.replace(/'/g, ''));
};

app.get('/api/llm/options', async (req, res) => {
  try {
    const sql = `
            SELECT COLUMN_NAME, COLUMN_TYPE 
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = ? 
            AND TABLE_NAME = 'LLM_DETAILS' 
            AND COLUMN_NAME IN ('LLM_PROVIDER', 'LLM_PROVIDER_TYPE')
        `;
    const [rows] = await dbPool.query(sql, [process.env.DB_DATABASE]);

    const options = { providers: [], types: [] };
    rows.forEach(row => {
      if (row.COLUMN_NAME === 'LLM_PROVIDER') {
        const allProviders = parseEnum(row.COLUMN_TYPE);
        // Allow users to add keys for LLMs and speech services
        options.providers = allProviders.filter(p => ['GROQ', 'GEMINI', 'GOOGLE_TTS', 'ELEVENLABS', 'DEEPGRAM'].includes(p));
      } else if (row.COLUMN_NAME === 'LLM_PROVIDER_TYPE') {
        options.types = parseEnum(row.COLUMN_TYPE);
      }
    });
    res.json(options);
  } catch (e) {
    logger.error('Failed to get LLM options', { error: e.message });
    res.status(500).json({ error: 'Failed to retrieve LLM options' });
  }
});

app.get('/api/llm/keys', async (req, res) => {
  const { userId, firmId } = req.query;
  if (!userId || !firmId) {
    return res.status(400).json({ error: 'userId and firmId are required.' });
  }
  try {
    const sql = `
            SELECT ID, LLM_PROVIDER, LLM_PROVIDER_TYPE, API_KEY, STATUS 
            FROM LLM_DETAILS 
            WHERE USERID = ? AND FIRMID = ?
        `;
    const [rows] = await dbPool.query(sql, [userId, firmId]);
    res.json(rows);
  } catch (e) {
    logger.error('Failed to get API keys', { error: e.message, userId, firmId });
    res.status(500).json({ error: 'Failed to retrieve API keys.' });
  }
});

app.post('/api/llm/keys', async (req, res) => {
  const { userId, firmId, llmProvider, llmProviderType, apiKey } = req.body;
  if (!userId || !firmId || !llmProvider || !llmProviderType || !apiKey) {
    return res.status(400).json({ error: 'All fields are required to save an API key.' });
  }
  try {
    const sql = `
            INSERT INTO LLM_DETAILS (USERID, FIRMID, LLM_PROVIDER, LLM_PROVIDER_TYPE, API_KEY)
            VALUES (?, ?, ?, ?, ?)
        `;
    await dbPool.query(sql, [userId, firmId, llmProvider, llmProviderType, apiKey]);
    logger.info('API Key saved', { userId, firmId, provider: llmProvider });
    res.status(201).json({ message: 'API key saved successfully.' });
  } catch (e) {
    logger.error('Failed to save API key', { error: e.message, userId, firmId });
    res.status(500).json({ error: 'Failed to save API key.' });
  }
});

app.put('/api/llm/keys/:id', async (req, res) => {
  const { id } = req.params;
  const { userId, firmId, API_KEY, STATUS } = req.body;
  if (!userId || !firmId || !API_KEY || !STATUS) {
    return res.status(400).json({ error: 'User, Firm, API Key, and Status are required.' });
  }
  try {
    const sql = 'UPDATE LLM_DETAILS SET API_KEY = ?, STATUS = ? WHERE ID = ? AND USERID = ? AND FIRMID = ?';
    const [result] = await dbPool.query(sql, [API_KEY, STATUS, id, userId, firmId]);
    if (result.affectedRows > 0) {
      logger.info('API Key updated', { id, userId, firmId });
      res.status(200).json({ message: 'API key updated.' });
    } else {
      res.status(404).json({ error: 'API key not found or you do not have permission to edit it.' });
    }
  } catch (e) {
    logger.error('Failed to update API key', { error: e.message, id });
    res.status(500).json({ error: 'Failed to update API key.' });
  }
});


app.delete('/api/llm/keys/:id', async (req, res) => {
  const { id } = req.params;
  const { userId, firmId } = req.body;
  if (!userId || !firmId) {
    return res.status(400).json({ error: 'userId and firmId are required for deletion.' });
  }
  try {
    const sql = 'DELETE FROM LLM_DETAILS WHERE ID = ? AND USERID = ? AND FIRMID = ?';
    const [result] = await dbPool.query(sql, [id, userId, firmId]);
    if (result.affectedRows > 0) {
      logger.info('API Key deleted', { id, userId, firmId });
      res.status(200).json({ message: 'API key deleted successfully.' });
    } else {
      res.status(404).json({ error: 'API key not found or you do not have permission to delete it.' });
    }
  } catch (e) {
    logger.error('Failed to delete API key', { error: e.message, id });
    res.status(500).json({ error: 'Failed to delete API key.' });
  }
});


// ============================================================================
// 16) SHARING ENDPOINTS
// ============================================================================
app.get('/api/rag/shares/:ownerId', async (req, res) => {
  const { ownerId } = req.params;
  try {
    const allShares = await readShares();
    const ownerShares = {};

    // Iterate over each grantee in the shares data
    for (const granteeId in allShares) {
      // Filter the shares for this grantee to find ones that belong to the owner
      const sharesFromOwner = allShares[granteeId].filter(
        share => String(share.ownerId) === ownerId
      );

      // If shares from this owner exist for the current grantee, add them to the result
      if (sharesFromOwner.length > 0) {
        ownerShares[granteeId] = sharesFromOwner;
      }
    }

    res.json(ownerShares);

  } catch (e) {
    logger.error('Failed to get shares for owner', { error: e.message, ownerId });
    res.status(500).json({ error: 'Could not retrieve sharing information.' });
  }
});


app.post('/api/rag/share', async (req, res) => {
  const { ownerId, categoryName, granteeId } = req.body;
  if (!ownerId || !categoryName || !granteeId) {
    return res.status(400).json({ error: 'Owner, category, and grantee are required.' });
  }

  // SECURITY FIX: Do not allow sharing with oneself.
  if (String(ownerId) === String(granteeId)) {
    return res.status(400).json({ error: 'You cannot share a knowledge base with yourself.' });
  }

  try {
    const shares = await readShares();
    if (!shares[granteeId]) {
      shares[granteeId] = [];
    }

    // FIX: Prevent duplicate shares by ensuring type-safe comparison.
    // ownerId from request is a string, but is stored as a number.
    const numOwnerId = Number(ownerId);
    const alreadyExists = shares[granteeId].some(
      share => share.ownerId === numOwnerId && share.categoryName === categoryName
    );

    if (alreadyExists) {
      return res.status(409).json({ error: 'This knowledge base is already shared with this user.' });
    }

    shares[granteeId].push({ ownerId: numOwnerId, categoryName });
    await writeShares(shares);

    logger.info('RAG shared successfully', { owner: ownerId, category: categoryName, grantee: granteeId });
    res.status(201).json({ message: 'Knowledge base shared successfully.' });
  } catch (e) {
    logger.error('Failed to share RAG', { error: e.message });
    res.status(500).json({ error: 'Failed to share the knowledge base.' });
  }
});

app.delete('/api/rag/share', async (req, res) => {
  const { ownerId, categoryName, granteeId } = req.body;
  if (!ownerId || !categoryName || !granteeId) {
    return res.status(400).json({ error: 'Owner, category, and grantee are required to revoke access.' });
  }
  try {
    const shares = await readShares();
    if (shares[granteeId]) {
      // FIX: Ensure type-safe comparison for deletion.
      const numOwnerId = Number(ownerId);
      shares[granteeId] = shares[granteeId].filter(
        share => !(share.ownerId === numOwnerId && share.categoryName === categoryName)
      );
      if (shares[granteeId].length === 0) {
        delete shares[granteeId];
      }
    }
    await writeShares(shares);
    logger.info('RAG share revoked', { owner: ownerId, category: categoryName, grantee: granteeId });
    res.status(200).json({ message: 'Access revoked.' });
  } catch (e) {
    logger.error('Failed to revoke RAG share', { error: e.message });
    res.status(500).json({ error: 'Failed to revoke access.' });
  }
});



// ============================================================================
// 17) SERVER
// ============================================================================
app.get('/', (_req, res) => {
  res
    .status(200)
    .send(`<h1>DEV RAG System Backend is running.</h1><p>AI Server Target: ${AI_SERVER_URL}</p>`);
});

const PORT = process.env.PORT || 8351;
const HOST = '0.0.0.0';

const getLocalIp = () => {
  const nets = os.networkInterfaces();
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      if (net.family === 'IPv4' && !net.internal) return net.address;
    }
  }
  return 'not found';
};

app.listen(PORT, HOST, () => {
  logger.info('RAG backend server started.', {
    port: PORT,
    host: HOST,
    localUrl: `http://localhost:${PORT}`,
    lanUrl: `http://${getLocalIp()}:${PORT}`,
    aiServer: AI_SERVER_URL,
  });
});

