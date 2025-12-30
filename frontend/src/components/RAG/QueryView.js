import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import axios from 'axios';
import Cookies from 'js-cookie';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, User, Loader2, ArrowUp, Folder, FileText, Trash2, RefreshCw, UploadCloud, PlusSquare, AlertTriangle, Settings, X, Square, Play, Search, ShieldCheck, Star, FileEdit, BrainCircuit, MessageSquare, Save, ChevronDown, Mic, Database } from 'lucide-react';
import styles from './rag.styles.js';
import { RAG_BACKEND_URL } from './rag.utils';

// ==============================================================================
// Query Interface Components
// ==============================================================================
const QueryView = ({ currentUser }) => {
    const [categories, setCategories] = useState([]);
    const [personas, setPersonas] = useState([]);
    const [selectedCategory, setSelectedCategory] = useState(null);
    const [selectedPersona, setSelectedPersona] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!currentUser) return;
        setLoading(true);

        const fetchData = async () => {
            const catEndpoint = `${RAG_BACKEND_URL}/api/rag/viewable?userId=${currentUser.id}&userRole=${currentUser.role}&firmId=${currentUser.firmId}`;
            const personaEndpoint = `${RAG_BACKEND_URL}/api/personas?firm_id=${currentUser.firmId}`;
            try {
                const [catResp, personaResp] = await Promise.all([
                    axios.get(catEndpoint),
                    axios.get(personaEndpoint)
                ]);
                setCategories(catResp.data || []);
                const personaData = personaResp.data || [];
                setPersonas(personaData);
                if (personaData.length > 0 && !selectedPersona) {
                    setSelectedPersona(personaData[0]);
                }
            } catch (err) {
                console.error(`Failed to fetch initial data for role ${currentUser.role}`, err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [currentUser, selectedPersona]);

    if (loading) {
        return <div style={styles.loadingContainer}><Loader2 style={styles.spinner} size={24} /> Loading...</div>;
    }

    const handleSelectCategory = (category) => {
        if (category.type === 'sql_agent') {
            setSelectedCategory(category);
            // SQL Agent doesn't strictly need a persona, but we can set a dummy one if needed or leave null.
            // QueryInterface handles null persona if isSqlMode is true? Let's check. 
            // QueryInterface checks `if (selectedCategory && selectedPersona)`. 
            // We MUST set a selectedPersona or make QueryInterface conditional.
            // Let's set a fake persona for SQL Agent to satisfy the render condition.
            setSelectedPersona({ id: 'sql-agent', name: 'SQL Agent' });
            return;
        }

        if (category.personaId && personas.length > 0) {
            const autoSelectedPersona = personas.find(p => p.id === category.personaId);
            if (autoSelectedPersona) {
                setSelectedPersona(autoSelectedPersona);
            } else {
                console.warn(`Persona ID ${category.personaId} not found. Defaulting.`);
                if (personas.length > 0) setSelectedPersona(personas[0]);
            }
        } else if (personas.length > 0) {
            setSelectedPersona(personas[0]);
        } else {
            // Fallback if no personas exist at all
            setSelectedPersona({ id: 'default', name: 'Default Assistant' });
        }
        setSelectedCategory(category);
    };

    if (selectedCategory && selectedPersona) {
        return <QueryInterface
            currentUser={currentUser}
            owner={selectedCategory.owner}
            category={selectedCategory.name}
            selectedCategory={selectedCategory}
            persona={selectedPersona}
            personas={personas}
            onBack={() => setSelectedCategory(null)}
            onPersonaChange={setSelectedPersona}
            isVoiceQuery={false}
        />;
    }

    return (
        <div style={{ maxWidth: '1024px', margin: '0 auto' }}>
            <header style={styles.header}>
                <h2 style={styles.headerH2}>Choose a Knowledge Base</h2>
                <p style={styles.headerSubtitle}>Select a knowledge base to start a conversation.</p>
            </header>
            <CategorySelector categories={categories} onSelect={handleSelectCategory} />
        </div>
    );
};

const CategorySelector = ({ categories, onSelect }) => (
    <div style={styles.card}>
        <div style={styles.cardHeader}><Folder size={20} /> Select a Category</div>
        <div className="list-group">
            {/* SQL Agent Option */}
            <div className="list-group-item" onClick={() => onSelect({ name: 'SQL Agent', type: 'sql_agent', owner: 'system' })}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontWeight: 600, color: 'var(--primary)' }}>
                    <Database size={18} /> <span>SQL Agent</span>
                </div>
                <span className="list-group-arrow">&rarr;</span>
            </div>

            <div style={{ borderTop: '1px solid var(--border)', margin: '0.5rem 0' }}></div>

            {categories.length > 0 ? categories.map(c => (
                <div key={`${c.owner}-${c.name}`} className="list-group-item" onClick={() => onSelect(c)}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <Folder size={18} style={{ color: 'var(--muted-foreground)' }} />
                        {c.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>
                    <span className="list-group-arrow">&rarr;</span>
                </div>
            )) : <p style={styles.p}>No active knowledge bases found for your account.</p>}
        </div>
    </div>
);

const VoiceSettingsModal = ({
    open,
    onClose,
    allVoices,
    selectedVoiceCode,
    onSelectVoice,
    onPlayDemo,
    ttsProvider,
    setTtsProvider,
    sttProvider,
    setSttProvider,
    apiKeys
}) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [activeDemo, setActiveDemo] = useState(null);

    const handlePlayDemo = async (voice) => {
        setActiveDemo(voice.code);
        await onPlayDemo(voice);
        setActiveDemo(null);
    };

    const availableTtsProviders = useMemo(() => {
        const providers = [{ id: 'piper', name: 'Standard (Piper)' }];
        if (apiKeys.google) providers.push({ id: 'google', name: 'Google Cloud' });
        if (apiKeys.elevenlabs) providers.push({ id: 'elevenlabs', name: 'ElevenLabs' });
        if (apiKeys.deepgram) providers.push({ id: 'deepgram', name: 'Deepgram' });
        return providers;
    }, [apiKeys]);

    const availableSttProviders = useMemo(() => {
        const providers = [{ id: 'whisper', name: 'Standard (Whisper)' }];
        if (apiKeys.deepgram) providers.push({ id: 'deepgram', name: 'Deepgram' });
        return providers;
    }, [apiKeys]);

    const currentVoices = useMemo(() => {
        return allVoices[ttsProvider] || [];
    }, [allVoices, ttsProvider]);

    const filteredVoices = useMemo(() => {
        const lowerSearch = searchTerm.toLowerCase();
        return currentVoices.filter(voice => voice.name.toLowerCase().includes(lowerSearch));
    }, [currentVoices, searchTerm]);


    if (!open) return null;

    return (
        <div style={styles.modalOverlay}>
            <div style={styles.modalContent}>
                <div style={styles.modalHeader}>
                    <h2 style={styles.modalTitle}>Voice Settings</h2>
                    <button onClick={onClose} style={styles.modalCloseButton}><X size={24} /></button>
                </div>
                <div style={styles.modalBody}>
                    <div style={styles.voiceSettingsSection}>
                        <h3 style={styles.voiceSettingsHeader}>Text-to-Speech (TTS) Provider</h3>
                        <div style={styles.providerToggleContainer}>
                            {availableTtsProviders.map(p => (
                                <button
                                    key={p.id}
                                    onClick={() => setTtsProvider(p.id)}
                                    style={ttsProvider === p.id ? styles.providerButtonActive : styles.providerButton}
                                >
                                    {p.name}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div style={styles.voiceSettingsSection}>
                        <h3 style={styles.voiceSettingsHeader}>Speech-to-Text (STT) Provider</h3>
                        <div style={styles.providerToggleContainer}>
                            {availableSttProviders.map(p => (
                                <button
                                    key={p.id}
                                    onClick={() => setSttProvider(p.id)}
                                    style={sttProvider === p.id ? styles.providerButtonActive : styles.providerButton}
                                >
                                    {p.name}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div style={styles.searchWrapper}>
                        <Search size={20} style={styles.searchIcon} />
                        <input
                            type="text"
                            placeholder={`Search ${ttsProvider} voices...`}
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            style={styles.searchInput}
                        />
                    </div>

                    <div style={styles.voiceListContainer}>
                        {filteredVoices.length === 0 ? (
                            <p style={{ textAlign: 'center', color: 'var(--muted-foreground)' }}>
                                No voices found for {ttsProvider}. Make sure your API key is active.
                            </p>
                        ) : (
                            filteredVoices.map(voice => (
                                <div key={voice.code} style={styles.voiceItem}>
                                    <span style={{ flex: 1 }}>{voice.name} {voice.accent ? `(${voice.accent})` : ''}</span>
                                    <button onClick={() => handlePlayDemo(voice)} style={styles.playDemoButtonSmall} disabled={activeDemo === voice.code}>
                                        {activeDemo === voice.code ? <Loader2 style={styles.spinner} size={16} /> : <Play size={16} />}
                                    </button>
                                    <button onClick={() => { onSelectVoice(voice.code); onClose(); }} style={voice.code === selectedVoiceCode ? styles.selectButtonActive : styles.selectButton}>
                                        Select
                                    </button>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

const StarRating = ({ rating, onRating }) => {
    return (
        <div style={styles.starRatingContainer}>
            {[...Array(5)].map((_, index) => {
                const ratingValue = index + 1;
                return (
                    <button
                        key={ratingValue}
                        style={styles.starButton}
                        onClick={() => onRating(ratingValue)}
                    >
                        <Star
                            size={18}
                            fill={ratingValue <= rating ? 'var(--warning)' : 'none'}
                            stroke={ratingValue <= rating ? 'var(--warning)' : 'var(--border)'}
                        />
                    </button>
                );
            })}
        </div>
    );
};

const QueryInterface = ({ currentUser, owner, category, selectedCategory, persona, personas, onBack, onPersonaChange, isVoiceQuery }) => {
    const [chat, setChat] = useState([]);
    const [liveTranscript, setLiveTranscript] = useState([]);
    const [status, setStatus] = useState('idle');
    const [isVoiceMode, setIsVoiceMode] = useState(isVoiceQuery);
    const [isSqlMode, setIsSqlMode] = useState(selectedCategory?.type === 'sql_agent');
    const [isVoiceModalOpen, setVoiceModalOpen] = useState(false);
    const [error, setError] = useState('');

    // Voice settings state
    const [allVoices, setAllVoices] = useState({ piper: [], google: [], elevenlabs: [], deepgram: [] });
    const [apiKeys, setApiKeys] = useState({ google: null, elevenlabs: null, deepgram: null });
    const [ttsProvider, setTtsProvider] = useState(localStorage.getItem('RAG_TTS_PROVIDER') || 'piper');
    const [sttProvider, setSttProvider] = useState(localStorage.getItem('RAG_STT_PROVIDER') || 'whisper');
    const [selectedVoiceCode, setSelectedVoiceCode] = useState(localStorage.getItem('RAG_USER_VOICE_PREFERENCE') || 'en_GB-alan-medium');

    const sessionIdRef = useRef(`${currentUser.id}-${currentUser.role}-${category}`);
    const audioPlayerRef = useRef(null);
    const chatEndRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const streamRef = useRef(null);
    const abortControllerRef = useRef(null);

    const audioContextRef = useRef(null);
    const analyserRef = useRef(null);
    const silenceTimeoutRef = useRef(null);
    const animationFrameIdRef = useRef(null);

    const firmid = Cookies.get('firmid');

    // Fetch all API keys on mount
    useEffect(() => {
        const fetchKeys = async () => {
            if (!currentUser || !firmid) return;
            try {
                const res = await axios.get(`${RAG_BACKEND_URL}/api/llm/keys?userId=${currentUser.id}&firmId=${firmid}`);
                const keys = res.data || [];
                const activeKeys = {
                    google: keys.find(k => k.LLM_PROVIDER === 'GOOGLE_TTS' && k.STATUS === 'ACTIVE')?.API_KEY || null,
                    elevenlabs: keys.find(k => k.LLM_PROVIDER === 'ELEVENLABS' && k.STATUS === 'ACTIVE')?.API_KEY || null,
                    deepgram: keys.find(k => k.LLM_PROVIDER === 'DEEPGRAM' && k.STATUS === 'ACTIVE')?.API_KEY || null,
                };
                setApiKeys(activeKeys);
            } catch (err) {
                console.error("Failed to fetch API keys for voice services.", err);
            }
        };
        fetchKeys();
    }, [currentUser, firmid]);

    // Fetch voices for all providers
    useEffect(() => {
        const fetchAllVoices = async () => {
            const endpoints = {
                piper: '/api/voice/list-voices',
                google: `/api/voice/list-google-voices?firm_id=${firmid}`,
                elevenlabs: `/api/voice/list-elevenlabs-voices?firm_id=${firmid}`,
                deepgram: '/api/voice/list-deepgram-voices'
            };

            const newVoices = { piper: [], google: [], elevenlabs: [], deepgram: [] };

            if (endpoints.piper) {
                try {
                    const res = await axios.get(`${RAG_BACKEND_URL}${endpoints.piper}`);
                    newVoices.piper = res.data || [];
                } catch (e) { console.error('Failed to fetch Piper voices', e); }
            }
            if (apiKeys.google) {
                try {
                    const res = await axios.get(`${RAG_BACKEND_URL}${endpoints.google}`);
                    newVoices.google = res.data || [];
                } catch (e) { console.error('Failed to fetch Google voices', e); }
            }
            if (apiKeys.elevenlabs) {
                try {
                    const res = await axios.get(`${RAG_BACKEND_URL}${endpoints.elevenlabs}`);
                    newVoices.elevenlabs = res.data || [];
                } catch (e) { console.error('Failed to fetch ElevenLabs voices', e); }
            }
            if (apiKeys.deepgram) {
                try {
                    const res = await axios.get(`${RAG_BACKEND_URL}${endpoints.deepgram}`);
                    newVoices.deepgram = res.data || [];
                } catch (e) { console.error('Failed to fetch Deepgram voices', e); }
            }

            setAllVoices(newVoices);
        };

        fetchAllVoices();
    }, [apiKeys, firmid]);

    useEffect(() => {
        if (currentUser && category) {
            const fetchHistory = async () => {
                try {
                    const res = await axios.get(`${RAG_BACKEND_URL}/api/chat/history/${currentUser.id}/${currentUser.role}/${category}`);
                    if (res.data && res.data.length > 0) {
                        setChat(res.data);
                    } else {
                        setChat([]);
                    }
                } catch (err) {
                    console.error("Failed to fetch chat history", err);
                    setError("Could not load previous chat history.");
                }
            };
            fetchHistory();
        }
    }, [currentUser, category, owner, currentUser.role]);

    useEffect(() => {
        if (chat.length === 0) return;
        const saveHistory = async () => {
            try {
                await axios.post(`${RAG_BACKEND_URL}/api/chat/history/${currentUser.id}/${currentUser.role}/${category}`, chat);
            } catch (err) {
                console.error("Failed to save chat history", err);
            }
        };
        const debounceTimeout = setTimeout(saveHistory, 1000);
        return () => clearTimeout(debounceTimeout);
    }, [chat, currentUser, category, currentUser.role]);


    useEffect(() => {
        localStorage.setItem('RAG_USER_VOICE_PREFERENCE', selectedVoiceCode);
        localStorage.setItem('RAG_TTS_PROVIDER', ttsProvider);
        localStorage.setItem('RAG_STT_PROVIDER', sttProvider);
    }, [selectedVoiceCode, ttsProvider, sttProvider]);

    const handleAudioEnd = useCallback(() => {
        setStatus(prevStatus => {
            if (prevStatus === 'speaking' || prevStatus === 'greeting') {
                return 'listening';
            }
            return prevStatus;
        });
    }, []);

    const handleRatingChange = (messageIndex, newRating) => {
        setChat(currentChat =>
            currentChat.map((msg, index) =>
                index === messageIndex ? { ...msg, rating: newRating } : msg
            )
        );
    };

    useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [chat, liveTranscript]);

    const stopSpeaking = useCallback(() => {
        if (audioPlayerRef.current) {
            audioPlayerRef.current.pause();
            audioPlayerRef.current.src = '';
        }
    }, []);

    const handleStop = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
        setStatus(isVoiceMode ? 'listening' : 'idle');
    }, [isVoiceMode]);

    const sendQuery = useCallback(async (text) => {
        if (!text || !text.trim()) {
            if (isVoiceMode) setStatus('listening');
            return;
        }

        stopSpeaking();
        setChat(prev => [...prev, { sender: 'user', text }]);
        setLiveTranscript(prev => [...prev, { sender: 'user', text }]);
        setError('');
        setStatus('thinking');

        const controller = new AbortController();
        abortControllerRef.current = controller;

        try {
            let answer, sources;

            if (isSqlMode) {
                const sqlPayload = {
                    query: text,
                    user_id: currentUser.id,
                    firm_id: firmid,
                    session_id: sessionIdRef.current
                };

                // 1. Generate SQL
                const genResp = await axios.post(`${RAG_BACKEND_URL}/api/sql_agent/generate`, sqlPayload, { signal: controller.signal });

                if (genResp.data.success && genResp.data.sql) {
                    const generatedSql = genResp.data.sql;

                    answer = `**Generated SQL:**\n\`\`\`sql\n${generatedSql}\n\`\`\``;

                } else {
                    throw new Error(genResp.data.error || "Failed to generate SQL from your question.");
                }
                sources = [];
            } else {
                const payload = {
                    owner_id: owner,
                    category,
                    question: text,
                    queried_by_id: currentUser.id,
                    queried_by_role: currentUser.role,
                    session_id: sessionIdRef.current,
                    persona_id: persona.id,
                    firmId: firmid,
                    query_source: isVoiceMode ? 'voice' : 'text'
                };
                const resp = await axios.post(`${RAG_BACKEND_URL}/api/rag/query`, payload, { signal: controller.signal });
                answer = resp.data.answer;
                sources = resp.data.sources;
            }

            setChat(prev => [...prev, { sender: 'ai', text: answer, sources, rating: 0 }]);
            setLiveTranscript(prev => [...prev, { sender: 'ai', text: answer }]);

            if (isVoiceMode) {
                if (answer) {
                    setStatus('speaking');
                    const voiceDetails = (allVoices[ttsProvider] || []).find(v => v.code === selectedVoiceCode);

                    const ttsPayload = {
                        text: answer,
                        code: selectedVoiceCode,
                        provider: ttsProvider,
                        firm_id: firmid,
                        language: voiceDetails?.language, // For Google TTS
                    };
                    const ttsResp = await axios.post(`${RAG_BACKEND_URL}/api/voice/tts`, ttsPayload, { responseType: 'blob' });

                    const audioUrl = URL.createObjectURL(ttsResp.data);
                    if (audioPlayerRef.current) {
                        audioPlayerRef.current.src = audioUrl;
                        audioPlayerRef.current.play().catch(e => {
                            console.error("Audio playback failed:", e);
                            setError("Audio playback failed. Please interact via text.");
                            setStatus('listening');
                        });
                    } else {
                        setStatus('listening');
                    }
                } else {
                    setStatus('listening');
                }
            } else {
                setStatus('idle');
            }
        } catch (err) {
            if (axios.isCancel(err)) {
                console.log("Request canceled by user.");
                setStatus(isVoiceMode ? 'listening' : 'idle');
                return;
            }
            const msg = err.response?.data?.error || 'Failed to get an answer.';
            setError(msg);
            const errorMsg = { sender: 'ai', text: `Error: ${msg}`, rating: 0 };
            setChat(prev => [...prev, errorMsg]);
            setLiveTranscript(prev => [...prev, errorMsg]);
            setStatus(isVoiceMode ? 'listening' : 'idle');
        } finally {
            abortControllerRef.current = null;
        }
    }, [owner, category, isVoiceMode, isSqlMode, selectedVoiceCode, stopSpeaking, currentUser.id, currentUser.role, persona.id, firmid, allVoices, ttsProvider]);

    const sendQueryRef = useRef(sendQuery);
    useEffect(() => { sendQueryRef.current = sendQuery; }, [sendQuery]);

    const stopAnalyzing = useCallback(() => {
        if (silenceTimeoutRef.current) { clearTimeout(silenceTimeoutRef.current); silenceTimeoutRef.current = null; }
        if (animationFrameIdRef.current) { cancelAnimationFrame(animationFrameIdRef.current); animationFrameIdRef.current = null; }
    }, []);

    const startAnalyzing = useCallback(() => {
        if (!analyserRef.current) return;
        const analyser = analyserRef.current;
        const dataArray = new Uint8Array(analyser.fftSize);
        const analyze = () => {
            analyser.getByteTimeDomainData(dataArray);
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) { sum += Math.pow((dataArray[i] / 128.0) - 1, 2); }
            const rms = Math.sqrt(sum / dataArray.length);
            if (rms > 0.02) {
                if (silenceTimeoutRef.current) { clearTimeout(silenceTimeoutRef.current); silenceTimeoutRef.current = null; }
            } else {
                if (!silenceTimeoutRef.current) {
                    silenceTimeoutRef.current = setTimeout(() => {
                        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
                            mediaRecorderRef.current.stop();
                        }
                        stopAnalyzing();
                    }, 1500);
                }
            }
            animationFrameIdRef.current = requestAnimationFrame(analyze);
        };
        animationFrameIdRef.current = requestAnimationFrame(analyze);
    }, [stopAnalyzing]);

    const handleExitVoiceMode = useCallback(() => {
        stopSpeaking();
        stopAnalyzing();
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') { mediaRecorderRef.current.stop(); }
        if (streamRef.current) { streamRef.current.getTracks().forEach(track => track.stop()); streamRef.current = null; }
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') { audioContextRef.current.close().catch(console.error); audioContextRef.current = null; }
        mediaRecorderRef.current = null;
        setIsVoiceMode(false);
        setStatus('idle');
    }, [stopSpeaking, stopAnalyzing]);

    useEffect(() => {
        if (status === 'listening' && mediaRecorderRef.current && mediaRecorderRef.current.state !== 'recording') {
            audioChunksRef.current = [];
            mediaRecorderRef.current.start();
            startAnalyzing();
        } else if (status !== 'listening') {
            stopAnalyzing();
        }
        return () => stopAnalyzing();
    }, [status, startAnalyzing, stopAnalyzing]);

    const prepareRecorder = useCallback(async () => {
        try {
            if (streamRef.current) { streamRef.current.getTracks().forEach(track => track.stop()); }
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;
            const context = new (window.AudioContext || window.webkitAudioContext)();
            audioContextRef.current = context;
            const source = context.createMediaStreamSource(stream);
            const analyser = context.createAnalyser();
            analyser.fftSize = 2048;
            analyserRef.current = analyser;
            source.connect(analyser);
            mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            audioChunksRef.current = [];
            mediaRecorderRef.current.ondataavailable = e => audioChunksRef.current.push(e.data);
            mediaRecorderRef.current.onstop = async () => {
                stopAnalyzing();
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                if (audioBlob.size < 3000) {
                    console.log(`Audio blob too small (${audioBlob.size} bytes), ignoring.`);
                    setStatus('listening');
                    return;
                }

                setStatus('thinking');
                const formData = new FormData();
                formData.append('audio', audioBlob, 'rec.webm');
                formData.append('provider', sttProvider);
                formData.append('firm_id', firmid);

                try {
                    // Ensure this uses the correct endpoint for STT
                    // Assuming RAG_BACKEND_URL is the base for backend
                    const sttResp = await axios.post(`${RAG_BACKEND_URL}/api/voice/stt`, formData);
                    if (sttResp.data.text && sttResp.data.text.trim()) {
                        await sendQueryRef.current(sttResp.data.text);
                    } else {
                        setStatus('listening');
                    }
                } catch (err) {
                    setError("Sorry, I had trouble transcribing that.");
                    setStatus('listening');
                }
            };
            return true;
        } catch (err) {
            setError("Microphone access denied. Please enable it in browser settings.");
            setIsVoiceMode(false);
            setStatus('idle');
            return false;
        }
    }, [stopAnalyzing, sttProvider, firmid]);

    const enterVoiceMode = useCallback(async () => {
        setLiveTranscript([]);
        setError('');
        setIsVoiceMode(true);
        setStatus('preparing');
        const ready = await prepareRecorder();
        if (!ready) return;
        setStatus('greeting');
        try {
            const voiceDetails = (allVoices[ttsProvider] || []).find(v => v.code === selectedVoiceCode);

            const greetingPayload = {
                code: selectedVoiceCode,
                persona_id: persona.id,
                firmId: firmid,
                provider: ttsProvider,
                language: voiceDetails?.language,
            };
            const ttsResp = await axios.post(`${RAG_BACKEND_URL}/api/voice/greeting`, greetingPayload, { responseType: 'blob' });

            const audioUrl = URL.createObjectURL(ttsResp.data);
            if (audioPlayerRef.current) {
                audioPlayerRef.current.src = audioUrl;
                audioPlayerRef.current.play().catch(e => { console.error("Greeting audio playback failed:", e); setStatus('listening'); });
            }
        } catch (err) { setError("Could not start voice mode."); setStatus('listening'); }
    }, [prepareRecorder, selectedVoiceCode, persona.id, firmid, ttsProvider, allVoices]);

    const handlePlayDemo = useCallback(async (voice) => {
        try {
            const payload = {
                code: voice.code,
                firmId: firmid,
                provider: ttsProvider,
                language: voice.language,
            };
            const resp = await axios.post(`${RAG_BACKEND_URL}/api/voice/demo`, payload, { responseType: 'blob' });
            const audioUrl = URL.createObjectURL(resp.data);
            const demoAudio = new Audio(audioUrl);
            demoAudio.play();
        } catch (err) { console.error('Failed to play voice demo', err); setError('Could not play voice preview.'); }
    }, [firmid, ttsProvider]);

    const handleNewChat = () => {
        setChat([]);
        sessionIdRef.current = `${currentUser.id}-${currentUser.role}-${category}-${Date.now()}`;
        setLiveTranscript([]);
    };

    const handleDeleteHistory = async () => {
        if (!window.confirm("Are you sure you want to delete this chat history? This cannot be undone.")) return;
        try {
            await axios.delete(`${RAG_BACKEND_URL}/api/chat/history/${currentUser.id}/${currentUser.role}/${category}`);
            handleNewChat();
        } catch (err) {
            console.error("Failed to delete history", err);
            setError("Failed to delete history.");
        }
    };

    const statusTextMap = { preparing: "Connecting...", greeting: "...", listening: "Listening...", thinking: "Thinking...", speaking: "Speaking..." };

    return (
        <div className="gemini-chat-view" style={isVoiceMode ? styles.voiceModeContainer : styles.chatContainer}>
            <audio ref={audioPlayerRef} onEnded={handleAudioEnd} hidden />
            <VoiceSettingsModal
                open={isVoiceModalOpen}
                onClose={() => setVoiceModalOpen(false)}
                allVoices={allVoices}
                selectedVoiceCode={selectedVoiceCode}
                onSelectVoice={setSelectedVoiceCode}
                onPlayDemo={handlePlayDemo}
                ttsProvider={ttsProvider}
                setTtsProvider={setTtsProvider}
                sttProvider={sttProvider}
                setSttProvider={setSttProvider}
                apiKeys={apiKeys}
            />

            {isVoiceMode ? (
                <div style={styles.voiceFullScreen}>
                    <div style={styles.liveTranscriptContainer}>
                        <div style={styles.chatHistoryContent}>
                            {liveTranscript.map((m, i) => (
                                <div key={i} style={m.sender === 'user' ? styles.userMessage : styles.aiMessage}>
                                    <div style={styles.messageAvatar}>{m.sender === 'user' ? <User size={20} /> : <Bot size={20} />}</div>
                                    <div style={m.sender === 'user' ? styles.userMessageContent : styles.aiMessageContent} className="message-content">
                                        {m.sender === 'ai' ? <ReactMarkdown children={m.text} remarkPlugins={[remarkGfm]} /> : m.text}
                                    </div>
                                </div>
                            ))}
                            {status === 'thinking' && (
                                <div style={styles.aiMessage}>
                                    <div style={styles.messageAvatar}><Bot size={20} /></div>
                                    <div style={styles.aiMessageContent}><Loader2 style={styles.spinner} size={20} /></div>
                                </div>
                            )}
                            <div ref={chatEndRef} />
                        </div>
                    </div>
                    <div style={styles.voiceStatusText}>{statusTextMap[status] || "Starting..."}</div>
                    <div style={status === 'listening' ? { ...styles.voiceMicIcon, ...styles.voiceMicIconListening } : styles.voiceMicIcon}><Mic size={40} /></div>
                    <button onClick={stopSpeaking} style={status === 'speaking' ? styles.voiceStopButton : { ...styles.voiceStopButton, opacity: 0, pointerEvents: 'none' }} title="Stop Speaking"><Square size={28} /></button>
                    <button onClick={handleExitVoiceMode} style={styles.voiceExitButton} title="Exit Voice Mode"><X size={24} /></button>
                </div>
            ) : (
                <>
                    <div style={styles.chatHeader}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flex: 1 }}>
                            <button onClick={onBack} className="backButton" style={styles.backButton} title="Back to Category Selection">&larr; Back</button>
                            <div style={styles.chatHeaderInfo}>
                                <div style={styles.chatHeaderInfoChip}><Folder size={14} /> {category.replace(/_/g, ' ')}</div>
                                {selectedCategory.personaId ? (
                                    <div style={styles.chatHeaderInfoChip} title="Persona automatically assigned by admin">
                                        <BrainCircuit size={14} /> {persona.name}
                                    </div>
                                ) : (
                                    <div style={styles.personaDropdownWrapper}>
                                        <BrainCircuit size={14} />
                                        <select
                                            value={persona.id}
                                            onChange={(e) => {
                                                const newPersona = personas.find(p => p.id === e.target.value);
                                                if (newPersona) onPersonaChange(newPersona);
                                            }}
                                            style={styles.personaDropdown}
                                        >
                                            {personas.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                                        </select>
                                        <ChevronDown size={16} style={styles.personaDropdownIcon} />
                                    </div>
                                )}
                            </div>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <button onClick={handleNewChat} className="icon-button" style={styles.iconButton} title="New Chat (Clear Screen)">
                                <PlusSquare size={20} />
                            </button>
                            <button onClick={handleDeleteHistory} className="icon-button" style={{ ...styles.iconButton, color: 'var(--danger)' }} title="Delete History">
                                <Trash2 size={20} />
                            </button>
                            <button onClick={() => setVoiceModalOpen(true)} className="icon-button" style={styles.iconButton} title="Voice Settings">
                                <Settings size={20} />
                            </button>
                        </div>
                    </div>
                    <div style={styles.chatHistory}>
                        <div style={styles.chatHistoryContent}>
                            {chat.length === 0 && <div style={styles.emptyChat}><Bot size={48} /><h2>How can I help you?</h2><p>Ask a question about the content in '{category}' while I act as '{persona.name}'.</p></div>}
                            {chat.map((m, i) => {
                                const uniqueSources = m.sources?.length > 0 ? [...new Set(m.sources.map(s => s.source.split('/').pop()))] : [];
                                return (
                                    <div key={i} style={m.sender === 'user' ? styles.userMessage : styles.aiMessage}>
                                        <div style={styles.messageAvatar}>{m.sender === 'user' ? <User size={20} /> : <Bot size={20} />}</div>
                                        <div style={m.sender === 'user' ? styles.userMessageContent : styles.aiMessageContent} className="message-content">
                                            <ReactMarkdown children={m.text} remarkPlugins={[remarkGfm]} />
                                            {uniqueSources.length > 0 &&
                                                <div style={styles.sourcesContainer}>
                                                    <strong>Sources:</strong> {uniqueSources.join(', ')}
                                                </div>
                                            }
                                            {m.sender === 'ai' && (
                                                <StarRating
                                                    rating={m.rating || 0}
                                                    onRating={(newRating) => handleRatingChange(i, newRating)}
                                                />
                                            )}
                                        </div>
                                    </div>
                                );
                            })}
                            {status === 'thinking' && <div style={styles.aiMessage}><div style={styles.messageAvatar}><Bot size={20} /></div><div style={styles.aiMessageContent}><Loader2 style={styles.spinner} size={20} /></div></div>}
                            <div ref={chatEndRef} />
                        </div>
                    </div>
                    <div style={styles.chatInputContainer}>
                        <div style={styles.chatInputArea}>
                            {error && <div style={{ ...styles.alert, ...styles.alertDanger, marginBottom: '1rem' }}>{error}</div>}
                            <ChatInput
                                status={status}
                                onSubmit={sendQuery}
                                onVoiceClick={enterVoiceMode}
                                onStop={handleStop}
                                isSqlMode={isSqlMode}
                                onToggleSqlMode={() => setIsSqlMode(!isSqlMode)}
                            />
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};

const ChatInput = ({ status, onSubmit, onVoiceClick, onStop, isSqlMode, onToggleSqlMode }) => {
    const [query, setQuery] = useState('');
    const [tables, setTables] = useState([]);
    const [filteredTables, setFilteredTables] = useState([]);
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [selectedIndex, setSelectedIndex] = useState(-1);
    const textareaRef = useRef(null);
    const suggestionsRef = useRef(null);
    const firmid = Cookies.get('firmid');

    // Fetch table names when SQL mode is enabled
    useEffect(() => {
        if (isSqlMode && firmid) {
            const fetchTables = async () => {
                try {
                    console.log('Fetching tables for firm_id:', firmid);
                    const res = await axios.get(`${RAG_BACKEND_URL}/api/sql_agent/tables?firm_id=${firmid}`);
                    console.log('Tables response:', res.data);
                    if (res.data.success && res.data.tables) {
                        setTables(res.data.tables);
                        console.log('Loaded tables:', res.data.tables);
                    }
                } catch (err) {
                    console.error('Failed to fetch table names:', err);
                }
            };
            fetchTables();
        } else {
            console.log('Not fetching tables - isSqlMode:', isSqlMode, 'firmid:', firmid);
        }
    }, [isSqlMode, firmid]);

    // Filter tables based on current input
    useEffect(() => {
        console.log('Filter effect - isSqlMode:', isSqlMode, 'query:', query, 'tables:', tables.length);

        if (!isSqlMode) {
            setFilteredTables([]);
            setShowSuggestions(false);
            return;
        }

        // If no query or only whitespace, show all tables
        if (!query.trim()) {
            if (tables.length > 0) {
                setFilteredTables(tables.slice(0, 10));
                setShowSuggestions(true);
                console.log('Showing all tables (no query)');
            } else {
                setFilteredTables([]);
                setShowSuggestions(false);
            }
            return;
        }

        // Get the word being typed (last word in the query)
        const words = query.split(/\s+/);
        const currentWord = words[words.length - 1].toLowerCase();

        if (currentWord.length > 0) {
            const matches = tables.filter(table =>
                table.toLowerCase().includes(currentWord)
            ).slice(0, 10); // Limit to 10 suggestions

            console.log('Filtered matches for "' + currentWord + '":', matches);
            setFilteredTables(matches);
            setShowSuggestions(matches.length > 0);
            setSelectedIndex(-1);
        } else {
            // Show all tables if current word is empty (e.g., after a space)
            setFilteredTables(tables.slice(0, 10));
            setShowSuggestions(tables.length > 0);
        }
    }, [query, tables, isSqlMode]);

    useEffect(() => {
        const el = textareaRef.current;
        if (el) {
            el.style.height = 'auto';
            el.style.height = `${el.scrollHeight}px`;
        }
    }, [query]);

    const insertTableName = (tableName) => {
        const words = query.split(/\s+/);
        words[words.length - 1] = tableName;
        const newQuery = words.join(' ') + ' ';
        setQuery(newQuery);
        setShowSuggestions(false);
        setSelectedIndex(-1);
        textareaRef.current?.focus();
    };

    const submitText = () => {
        if (!query.trim() || status !== 'idle') return;
        onSubmit(query);
        setQuery('');
        setShowSuggestions(false);
    };

    const handleKeyDown = (e) => {
        if (showSuggestions && filteredTables.length > 0) {
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                setSelectedIndex(prev =>
                    prev < filteredTables.length - 1 ? prev + 1 : prev
                );
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                setSelectedIndex(prev => prev > 0 ? prev - 1 : -1);
            } else if (e.key === 'Enter' && selectedIndex >= 0) {
                e.preventDefault();
                insertTableName(filteredTables[selectedIndex]);
                return;
            } else if (e.key === 'Escape') {
                setShowSuggestions(false);
                setSelectedIndex(-1);
                return;
            }
        }

        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitText();
        }
    };

    const renderButton = () => {
        if (status === 'thinking') {
            return (<button onClick={onStop} style={{ ...styles.sendButton, background: 'var(--danger)' }} aria-label="Stop generation"><Square size={20} /></button>);
        }
        if (query.trim()) {
            return (<button onClick={submitText} style={status !== 'idle' ? { ...styles.sendButton, ...styles.sendButtonDisabled } : styles.sendButton} disabled={status !== 'idle'} aria-label="Send message"><ArrowUp size={20} /></button>);
        }
        return (<button onClick={onVoiceClick} style={status !== 'idle' ? { ...styles.sendButton, ...styles.sendButtonDisabled } : styles.sendButton} disabled={status !== 'idle'} aria-label="Start voice conversation"><Mic size={20} /></button>);
    };

    return (
        <div style={{ position: 'relative' }}>
            <div style={styles.inputWrapper}>
                <button
                    onClick={onToggleSqlMode}
                    style={{
                        ...styles.iconButton,
                        marginRight: '0.5rem',
                        color: isSqlMode ? 'var(--primary)' : 'var(--muted-foreground)',
                        backgroundColor: isSqlMode ? 'rgba(var(--primary-rgb), 0.1)' : 'transparent',
                        position: 'relative'
                    }}
                    title={isSqlMode ? `SQL Agent (${tables.length} tables loaded)` : "Enable SQL Agent"}
                >
                    <Database size={20} />
                    {isSqlMode && tables.length > 0 && (
                        <span style={{
                            position: 'absolute',
                            top: '-4px',
                            right: '-4px',
                            background: 'var(--primary)',
                            color: 'white',
                            borderRadius: '50%',
                            width: '16px',
                            height: '16px',
                            fontSize: '10px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontWeight: 'bold'
                        }}>
                            {tables.length}
                        </span>
                    )}
                </button>
                {isSqlMode && tables.length > 0 && (
                    <button
                        onClick={() => {
                            setFilteredTables(tables.slice(0, 10));
                            setShowSuggestions(true);
                        }}
                        style={{
                            ...styles.iconButton,
                            marginRight: '0.5rem',
                            fontSize: '11px',
                            padding: '0.3rem 0.6rem',
                            borderRadius: '12px',
                            background: 'var(--secondary)',
                            border: '1px solid var(--border)'
                        }}
                        title="Show all tables"
                    >
                        Show Tables
                    </button>
                )}
                <textarea
                    ref={textareaRef}
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    style={styles.chatInput}
                    placeholder={isSqlMode ? "Ask a question or start typing table names..." : "Ask a question..."}
                    disabled={status === 'thinking'}
                    rows={1}
                />
                {renderButton()}
            </div>

            {/* Autocomplete Suggestions Dropdown */}
            {showSuggestions && filteredTables.length > 0 && (
                <div ref={suggestionsRef} style={styles.autocompleteDropdown}>
                    <div style={styles.autocompleteHeader}>
                        <Database size={14} />
                        <span>Available Tables</span>
                    </div>
                    {filteredTables.map((table, index) => (
                        <div
                            key={table}
                            style={{
                                ...styles.autocompleteItem,
                                ...(index === selectedIndex ? styles.autocompleteItemSelected : {})
                            }}
                            onClick={() => insertTableName(table)}
                            onMouseEnter={() => setSelectedIndex(index)}
                        >
                            {table}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};


export default QueryView;
