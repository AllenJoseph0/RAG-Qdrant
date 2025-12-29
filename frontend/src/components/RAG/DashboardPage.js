import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import Cookies from 'js-cookie';
import { Folder, BrainCircuit, ShieldAlert, Key, UploadCloud, Loader2, FileText, PlusSquare, RefreshCw, Trash2, AlertTriangle, FileEdit, Size, Save, X, Database, Download, Edit3, Search } from 'lucide-react';
import SqlAgentManager from './SqlAgentManager';
import styles from './rag.styles.js';
import { RAG_BACKEND_URL, generateUUID, formatBytes, formatDate, formatTime } from './rag.utils';
import {
    CategoryAccessControl,
    RedactionRulebook,
    PersonaSelector,
    ComplianceSelector,
    ComplianceTester,
    ApiKeyManager,
    SharingManager
} from './AdminTools';


// ==============================================================================
// File Management Modal (New)
// ==============================================================================
const FileManagementModal = ({ category, files = [], username, onClose, onFileDeleted }) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [descriptions, setDescriptions] = useState({});
    const [editingDesc, setEditingDesc] = useState(null); // filename
    const [descValue, setDescValue] = useState('');
    const [isSaving, setIsSaving] = useState(false);
    const [downloadingZip, setDownloadingZip] = useState(false);

    // Filter files
    const filteredFiles = files.filter(f => {
        const fname = typeof f === 'string' ? f : f.name;
        return fname.toLowerCase().includes(searchTerm.toLowerCase());
    });

    // Fetch descriptions
    useEffect(() => {
        const fetchDescriptions = async () => {
            try {
                const res = await axios.get(`${RAG_BACKEND_URL}/api/rag/files/descriptions?username=${username}&category=${category}`);
                setDescriptions(res.data || {});
            } catch (err) {
                // Silent fail
            }
        };
        if (category && username) fetchDescriptions();
    }, [category, username]);

    const handleDownload = (filename) => {
        const link = document.createElement('a');
        link.href = `${RAG_BACKEND_URL}/api/rag/files/download?username=${username}&category=${category}&filename=${encodeURIComponent(filename)}`;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const handleDownloadZip = async () => {
        setDownloadingZip(true);
        try {
            const response = await axios.get(`${RAG_BACKEND_URL}/api/rag/files/download-zip?username=${username}&category=${category}`, {
                responseType: 'blob'
            });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `${category}_files.zip`);
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (err) {
            alert("Failed to download ZIP.");
        } finally {
            setDownloadingZip(false);
        }
    };

    const handleDelete = async (filename) => {
        if (!window.confirm(`Delete ${filename}?`)) return;
        try {
            await axios.delete(`${RAG_BACKEND_URL}/api/rag/files/delete`, {
                data: { username, category, filename }
            });
            onFileDeleted();
        } catch (err) {
            alert("Failed to delete file.");
        }
    };

    const startEditDesc = (filename) => {
        setEditingDesc(filename);
        setDescValue(descriptions[filename] || '');
    };

    const saveDesc = async (filename) => {
        setIsSaving(true);
        try {
            await axios.post(`${RAG_BACKEND_URL}/api/rag/files/description`, {
                username, category, filename, description: descValue
            });
            setDescriptions(prev => ({ ...prev, [filename]: descValue }));
            setEditingDesc(null);
        } catch (err) {
            alert("Failed to save description");
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <div style={styles.modalOverlay}>
            <div style={styles.modalContent}>
                <div style={styles.modalHeader}>
                    <div>
                        <h2 style={styles.modalTitle}>Manage Knowledge Base</h2>
                        <p style={{ margin: 0, color: 'var(--muted-foreground)', fontSize: '0.9rem' }}>Category: {category}</p>
                    </div>
                    <button onClick={onClose} style={styles.modalCloseButton}><X size={24} /></button>
                </div>

                <div style={{ padding: '1.5rem 1.5rem 0.5rem' }}>
                    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                        <div style={{ ...styles.searchWrapper, flex: 1, margin: 0 }}>
                            <Search size={18} style={styles.searchIcon} />
                            <input
                                style={styles.searchInput}
                                placeholder="Search files..."
                                value={searchTerm}
                                onChange={e => setSearchTerm(e.target.value)}
                            />
                        </div>
                        <button onClick={handleDownloadZip} disabled={downloadingZip || files.length === 0} style={styles.buttonSecondary}>
                            {downloadingZip ? <Loader2 className="spin" size={16} /> : <Download size={16} />} Download All (Zip)
                        </button>
                    </div>
                </div>

                <div style={styles.modalBody}>
                    {filteredFiles.length === 0 ? <p style={{ textAlign: 'center', color: 'var(--muted-foreground)' }}>No files found.</p> : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                            {filteredFiles.map((file, idx) => {
                                const fname = typeof file === 'string' ? file : file.name;
                                const size = typeof file === 'string' ? 0 : file.size;
                                const date = typeof file === 'string' ? null : file.added;
                                const isEditing = editingDesc === fname;

                                return (
                                    <div key={idx} style={{
                                        border: '1px solid var(--border)',
                                        borderRadius: 'var(--radius)',
                                        padding: '1rem',
                                        background: 'var(--card)',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        gap: '0.75rem'
                                    }}>
                                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', overflow: 'hidden' }}>
                                                <div style={{
                                                    width: '40px', height: '40px', borderRadius: '8px',
                                                    background: 'rgba(59, 130, 246, 0.1)', color: 'var(--primary)',
                                                    display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0
                                                }}>
                                                    <FileText size={20} />
                                                </div>
                                                <div style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                                    <div style={{ fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={fname}>{fname}</div>
                                                    <div style={{ fontSize: '0.8rem', color: 'var(--muted-foreground)' }}>
                                                        {formatBytes(size)} • {date ? formatDate(date) : 'Unknown date'}
                                                    </div>
                                                </div>
                                            </div>

                                            <div style={{ display: 'flex', gap: '0.5rem', flexShrink: 0 }}>
                                                <button onClick={() => handleDownload(fname)} style={styles.iconButton} title="Download">
                                                    <Download size={18} />
                                                </button>
                                                <button onClick={() => startEditDesc(fname)} style={styles.iconButton} title="Edit Description">
                                                    <Edit3 size={18} />
                                                </button>
                                                <button onClick={() => handleDelete(fname)} style={{ ...styles.iconButton, color: 'var(--danger)', background: 'rgba(239, 68, 68, 0.1)' }} title="Delete">
                                                    <Trash2 size={18} />
                                                </button>
                                            </div>
                                        </div>

                                        {(descriptions[fname] || isEditing) && (
                                            <div style={{
                                                background: 'var(--secondary)',
                                                padding: '0.75rem',
                                                borderRadius: 'var(--radius-sm)',
                                                fontSize: '0.9rem'
                                            }}>
                                                {isEditing ? (
                                                    <div style={{ display: 'flex', gap: '0.5rem', flexDirection: 'column' }}>
                                                        <textarea
                                                            autoFocus
                                                            style={{ ...styles.input, minHeight: '60px', padding: '0.5rem', fontSize: '0.9rem' }}
                                                            value={descValue}
                                                            onChange={e => setDescValue(e.target.value)}
                                                            placeholder="Add a summary or description for this file..."
                                                        />
                                                        <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
                                                            <button onClick={() => setEditingDesc(null)} style={{ ...styles.buttonSecondary, padding: '0.3rem 0.8rem', fontSize: '0.8rem' }}>Cancel</button>
                                                            <button onClick={() => saveDesc(fname)} disabled={isSaving} style={{ ...styles.buttonPrimary, padding: '0.3rem 0.8rem', fontSize: '0.8rem' }}>
                                                                {isSaving ? 'Saving...' : 'Save'}
                                                            </button>
                                                        </div>
                                                    </div>
                                                ) : (
                                                    <div style={{ color: 'var(--foreground-heavy)', fontStyle: 'italic' }}>
                                                        {descriptions[fname]}
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>

                <div style={styles.modalFooter}>
                    <button onClick={onClose} style={styles.buttonSecondary}>Close</button>
                </div>
            </div>
        </div>
    );
};
const DashboardPage = ({ currentUser }) => {
    const [dashboardView, setDashboardView] = useState('knowledge');
    return (
        <div style={{ maxWidth: '1024px', margin: '0 auto', padding: '0 1rem', width: '100%' }}>
            <header style={styles.header}>
                <h2 style={styles.headerH2}>Admin Dashboard</h2>
                <p style={styles.headerSubtitle}>Manage your AI's knowledge, personas, and compliance rules.</p>
            </header>
            <div style={styles.dashboardNav}>
                <button onClick={() => setDashboardView('knowledge')} style={dashboardView === 'knowledge' ? styles.dashboardNavButtonActive : styles.dashboardNavButton}>
                    <Folder size={18} /> Knowledge Base
                </button>
                <button onClick={() => setDashboardView('personas')} style={dashboardView === 'personas' ? styles.dashboardNavButtonActive : styles.dashboardNavButton}>
                    <BrainCircuit size={18} /> Persona Engine
                </button>
                <button onClick={() => setDashboardView('compliance')} style={dashboardView === 'compliance' ? styles.dashboardNavButtonActive : styles.dashboardNavButton}>
                    <ShieldAlert size={18} /> Compliance
                </button>
                <button onClick={() => setDashboardView('apiKeys')} style={dashboardView === 'apiKeys' ? styles.dashboardNavButtonActive : styles.dashboardNavButton}>
                    <Key size={18} /> API Keys
                </button>
                <button onClick={() => setDashboardView('sqlAgent')} style={dashboardView === 'sqlAgent' ? styles.dashboardNavButtonActive : styles.dashboardNavButton}>
                    <Database size={18} /> SQL Agent
                </button>
                {/*
                 <button onClick={() => setDashboardView('share')} style={dashboardView === 'share' ? styles.dashboardNavButtonActive : styles.dashboardNavButton}>
                    <Share2 size={18}/> Sharing Manager
                </button>
                */}

            </div>
            {dashboardView === 'knowledge' && <KnowledgeBaseManager currentUser={currentUser} />}
            {dashboardView === 'personas' && <PersonaManager currentUser={currentUser} />}
            {dashboardView === 'compliance' && <ComplianceManager currentUser={currentUser} />}
            {dashboardView === 'apiKeys' && <ApiKeyManager currentUser={currentUser} />}
            {dashboardView === 'sqlAgent' && <SqlAgentManager currentUser={currentUser} />}
            {dashboardView === 'share' && <SharingManager currentUser={currentUser} />}
        </div>
    );
};

// ==============================================================================
// Knowledge Base Manager
// ==============================================================================
const KnowledgeBaseManager = ({ currentUser }) => {
    const [structure, setStructure] = useState([]);
    const [personas, setPersonas] = useState([]);
    const [complianceProfiles, setComplianceProfiles] = useState([]);
    const [loading, setLoading] = useState(true);
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');

    const [selectedCategory, setSelectedCategory] = useState('');
    const [newCategoryName, setNewCategoryName] = useState('');

    const [files, setFiles] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const [activeJob, setActiveJob] = useState(null);
    const [managingCategory, setManagingCategory] = useState(null);

    const fetchData = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            const fid = currentUser.firmId || Cookies.get('firmid');
            const [structureResp, personasResp, complianceResp] = await Promise.all([
                axios.get(`${RAG_BACKEND_URL}/api/rag/structure?username=${fid}`),
                axios.get(`${RAG_BACKEND_URL}/api/personas?firm_id=${fid}`),
                axios.get(`${RAG_BACKEND_URL}/api/compliance`)
            ]);
            setStructure(structureResp.data?.[fid] || []);
            setPersonas(personasResp.data || []);
            setComplianceProfiles(complianceResp.data || []);
        } catch (err) {
            setError(err.response?.data?.error || 'Could not fetch data.');
        } finally {
            setLoading(false);
        }
    }, [currentUser.id, currentUser.firmId]);

    useEffect(() => { fetchData(); }, [fetchData]);

    const handlePermissionsChange = (categoryName, newPermissions) => {
        setStructure(currentStructure =>
            currentStructure.map(cat =>
                cat.name === categoryName ? { ...cat, permissions: newPermissions } : cat
            )
        );
    };

    const handlePersonaChange = (categoryName, newPersonaId) => {
        setStructure(currentStructure =>
            currentStructure.map(cat =>
                cat.name === categoryName ? { ...cat, personaId: newPersonaId } : cat
            )
        );
    };

    const handleComplianceChange = (categoryName, newProfileId) => {
        setStructure(currentStructure =>
            currentStructure.map(cat =>
                cat.name === categoryName ? { ...cat, complianceProfileId: newProfileId } : cat
            )
        );
    };

    const handleAction = async (action, category) => {
        const confirmationText = {
            'update-index': `This will quickly update the index with new files for '${category}'. Continue?`,
            'create-index': `This will BUILD the entire index for '${category}'. This is a slow process required after deleting files. Continue?`,
            'delete-index': `This will delete the AI index for '${category}', but keep the uploaded files. Continue?`,
            'delete-category': `This will PERMANENTLY delete the category '${category}', all its files, and its index. This cannot be undone. Continue?`
        };
        if (!window.confirm(confirmationText[action])) return;

        setMessage(''); setError('');
        setActiveJob({ type: action, category });

        try {
            const payload = {
                username: currentUser.firmId || Cookies.get('firmid'),
                category,
                firm_id: currentUser.firmId
            };
            const resp = await axios.post(`${RAG_BACKEND_URL}/api/rag/${action}`, payload);
            setMessage(resp.data.message || 'Action completed successfully.');
            fetchData();
        } catch (err) {
            setError(err.response?.data?.error || `Failed to ${action.replace('-', ' ')}.`);
        } finally {
            setActiveJob(null);
        }
    };

    const handleUpload = async (e) => {
        e.preventDefault();
        const finalCategoryName = selectedCategory === '__NEW__' ? newCategoryName.trim() : selectedCategory;

        if (!files || !finalCategoryName) {
            setError('Please provide a category name and select files.');
            return;
        }
        const fd = new FormData();
        fd.append('username', currentUser.firmId || Cookies.get('firmid'));
        fd.append('category', finalCategoryName);
        for (let i = 0; i < files.length; i++) fd.append('files', files[i]);

        setIsUploading(true); setError(''); setMessage('');
        try {
            const resp = await axios.post(`${RAG_BACKEND_URL}/api/rag/upload`, fd);
            setMessage(`${resp.data.message} You should now build or update the index.`);
            setSelectedCategory('');
            setNewCategoryName('');
            setFiles(null);
            if (e.target.reset) e.target.reset();
            fetchData();
        } catch (err) {
            setError(err.response?.data?.error || 'File upload failed.');
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="fade-in">
            {message && <div style={{ ...styles.alert, ...styles.alertSuccess, marginTop: '1rem' }}>{message}</div>}
            {error && <div style={{ ...styles.alert, ...styles.alertDanger, marginTop: '1rem' }}>{error}</div>}

            <div style={styles.card}>
                <div style={styles.cardHeader}><UploadCloud size={20} /> Upload New Documents</div>
                <div style={styles.cardBody}>
                    <form onSubmit={handleUpload}>
                        <label style={styles.label}>1. Choose a Category</label>
                        <select
                            style={styles.input}
                            value={selectedCategory}
                            onChange={e => setSelectedCategory(e.target.value)}
                            required
                        >
                            <option value="" disabled>Select an existing category or create new</option>
                            {structure.map(cat => <option key={cat.name} value={cat.name}>{cat.name}</option>)}
                            <option value="__NEW__">-- Create a new category --</option>
                        </select>

                        {selectedCategory === '__NEW__' && (
                            <input
                                style={{ ...styles.input, marginTop: '1rem' }}
                                value={newCategoryName}
                                onChange={e => setNewCategoryName(e.target.value)}
                                placeholder="Enter new category name..."
                                required
                            />
                        )}

                        <label style={{ ...styles.label, marginTop: '1rem' }}>2. Choose Files to Upload</label>
                        <input type="file" multiple required onChange={e => setFiles(e.target.files)} style={styles.fileInput} />
                        <button type="submit" style={styles.buttonPrimary} disabled={isUploading}>
                            {isUploading ? <><Loader2 style={styles.spinner} size={16} /> Uploading...</> : 'Upload Files'}
                        </button>
                    </form>
                </div>
            </div>

            {loading && <div style={styles.loadingContainer}><Loader2 style={styles.spinner} size={24} /> Loading categories...</div>}

            {!loading && structure.length === 0 && <p style={styles.p}>No categories found. Upload documents to begin.</p>}

            {structure.map(cat => (
                <div key={cat.name} style={styles.card}>
                    <div style={styles.categoryHeader}>
                        <div style={styles.categoryTitle}>
                            <Folder size={20} /> Category: {cat.name}
                            <span style={{ ...styles.indexStatus, backgroundColor: cat.indexStatus === 'ACTIVE' ? 'var(--success)' : 'var(--warning-dark)' }}>
                                {cat.indexStatus}
                            </span>
                        </div>
                        <button onClick={() => setManagingCategory(cat)} style={styles.buttonSecondary}>
                            <Folder size={16} /> Manage Files ({cat.files.length})
                        </button>
                    </div>
                    {/* File list moved to modal */}
                    <div style={styles.cardFooter}>
                        <div style={styles.buttonGroup}>
                            <button style={styles.buttonSuccess} onClick={() => handleAction('update-index', cat.name)} disabled={!!activeJob} title="Fast: Add new files to the index.">
                                {activeJob?.type === 'update-index' && activeJob?.category === cat.name ? <><Loader2 style={styles.spinner} size={16} /> Updating</> : <><PlusSquare size={16} /> Update Index</>}
                            </button>
                            <button style={styles.buttonWarning} onClick={() => handleAction('create-index', cat.name)} disabled={!!activeJob} title="Slow: Re-process all files from scratch.">
                                {activeJob?.type === 'create-index' && activeJob?.category === cat.name ? <><Loader2 style={styles.spinner} size={16} /> Building</> : <><RefreshCw size={16} /> Build Index</>}
                            </button>
                            <button style={styles.buttonDangerOutline} onClick={() => handleAction('delete-category', cat.name)} disabled={!!activeJob} title="PERMANENTLY delete files and index.">
                                <AlertTriangle size={16} /> Delete Category
                            </button>
                        </div>
                    </div>
                    {currentUser.role === 'admin' && (
                        <>
                            <CategoryAccessControl
                                categoryName={cat.name}
                                initialPermissions={cat.permissions}
                                adminId={currentUser.firmId || Cookies.get('firmid')}
                                onPermissionsChange={handlePermissionsChange}
                                currentUserRole={currentUser.role}
                            />
                            <RedactionRulebook
                                adminId={currentUser.firmId || Cookies.get('firmid')}
                                categoryName={cat.name}
                            />
                            <PersonaSelector
                                categoryName={cat.name}
                                adminId={currentUser.firmId || Cookies.get('firmid')}
                                currentPersonaId={cat.personaId}
                                personas={personas}
                                onPersonaChange={handlePersonaChange}
                            />
                            <ComplianceSelector
                                categoryName={cat.name}
                                adminId={currentUser.firmId || Cookies.get('firmid')}
                                currentProfileId={cat.complianceProfileId}
                                profiles={complianceProfiles}
                                onProfileChange={handleComplianceChange}
                            />
                            <ComplianceTester
                                adminId={currentUser.firmId || Cookies.get('firmid')}
                                categoryName={cat.name}
                                personaId={cat.personaId}
                                complianceProfileId={cat.complianceProfileId}
                            />
                        </>
                    )}
                </div>
            ))}
            {managingCategory && (
                <FileManagementModal
                    category={managingCategory.name}
                    files={managingCategory.files}
                    username={currentUser.firmId || Cookies.get('firmid')}
                    onClose={() => setManagingCategory(null)}
                    onFileDeleted={() => {
                        fetchData();
                        // Optimistic update could go here but fetchData is safer
                        setManagingCategory(null); // Close to refresh or keep open? Ideally keep open but data might be stale.
                        // Actually, let's just close it or refetch and update local state in future.
                    }}
                />
            )}
        </div>
    );
};


// ==============================================================================
// Persona Manager Component
// ==============================================================================
const PersonaManager = ({ currentUser }) => {
    const [personas, setPersonas] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');
    const [editingPersona, setEditingPersona] = useState(null); // null or persona object
    const [isCreating, setIsCreating] = useState(false);

    const fetchPersonas = useCallback(async () => {
        setIsLoading(true);
        try {
            const fid = currentUser.firmId || Cookies.get('firmid');
            const res = await axios.get(`${RAG_BACKEND_URL}/api/personas?firm_id=${fid}`);
            setPersonas(res.data);
        } catch (err) {
            setError('Failed to fetch personas.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchPersonas();
    }, [fetchPersonas]);

    const handleSave = async (personaData) => {
        const isUpdate = !!personaData.id;
        const url = isUpdate ? `${RAG_BACKEND_URL}/api/personas/${personaData.id}` : `${RAG_BACKEND_URL}/api/personas`;
        const method = isUpdate ? 'put' : 'post';

        const payload = {
            ...personaData,
            firm_id: currentUser.firmId
        };

        try {
            await axios[method](url, payload);
            setEditingPersona(null);
            setIsCreating(false);
            fetchPersonas();
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to save persona.');
        }
    };

    const handleDelete = async (personaId) => {
        if (window.confirm('Are you sure you want to delete this persona?')) {
            try {
                await axios.delete(`${RAG_BACKEND_URL}/api/personas/${personaId}`);
                fetchPersonas();
            } catch (err) {
                setError('Failed to delete persona.');
            }
        }
    };

    const PersonaForm = ({ persona, onSave, onCancel }) => {
        const [name, setName] = useState(persona?.name || '');
        const [prompt, setPrompt] = useState(persona?.prompt || '');
        const [voicePrompt, setVoicePrompt] = useState(persona?.voice_prompt || '');
        const [stages, setStages] = useState((persona?.stages || []).join(', '));
        const [isSaving, setIsSaving] = useState(false);
        const isEditing = !!persona?.id;

        const handleSubmit = (e) => {
            e.preventDefault();
            if (!name.trim()) return;
            setIsSaving(true);
            const stagesArray = stages.split(',').map(s => s.trim()).filter(Boolean);
            onSave({ ...persona, name, prompt, voice_prompt: voicePrompt, stages: stagesArray }).finally(() => setIsSaving(false));
        };

        return (
            <div style={styles.modalOverlay}>
                <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
                    <div style={styles.modalHeader}>
                        <h2 style={styles.modalTitle}>{isEditing ? 'Edit Persona' : 'Create Persona'}</h2>
                        <button onClick={onCancel} style={styles.modalCloseButton}><X size={20} /></button>
                    </div>
                    <form onSubmit={handleSubmit} style={styles.modalBody}>
                        <div style={styles.formGroup}>
                            <label style={styles.label}>Persona Name</label>
                            <input className="modal-input" style={styles.input} value={name} onChange={e => setName(e.target.value)} placeholder="e.g., Customer Support Agent" required />
                            {!isEditing && <p style={styles.formHelperText}>If you leave the prompts blank, the AI will automatically generate them for you.</p>}
                        </div>
                        <div style={styles.promptContainer}>
                            <div style={{ ...styles.formGroup, ...styles.promptColumn }}>
                                <label style={styles.label}>System Prompt (for Voice Chat)</label>
                                <textarea className="modal-input" style={styles.textarea} value={voicePrompt} onChange={e => setVoicePrompt(e.target.value)} rows={10} placeholder="Define the AI's role, rules, and personality for voice-based interactions..." />
                            </div>
                            <div style={{ ...styles.formGroup, ...styles.promptColumn }}>
                                <label style={styles.label}>System Prompt (for Text Chat)</label>
                                <textarea className="modal-input" style={styles.textarea} value={prompt} onChange={e => setPrompt(e.target.value)} rows={10} placeholder="Define the AI's role, rules, and personality for text-based interactions..." />
                            </div>
                        </div>
                        <div style={styles.formGroup}>
                            <label style={styles.label}>Conversation Stages (comma-separated)</label>
                            <input className="modal-input" style={styles.input} value={stages} onChange={e => setStages(e.target.value)} placeholder="e.g., qualification, pitch, objection_handling" />
                        </div>
                    </form>
                    <div style={styles.modalFooter}>
                        <button type="button" onClick={onCancel} style={styles.buttonSecondary}>Cancel</button>
                        <button type="button" onClick={handleSubmit} style={styles.buttonSuccess} disabled={isSaving || !name.trim()}>
                            {isSaving ? <><Loader2 style={styles.spinner} size={16} /> Saving...</> : <><Save size={16} /> Save Persona</>}
                        </button>
                    </div>
                </div>
            </div>
        );
    };

    if (isLoading) return <div style={styles.loadingContainer}><Loader2 style={styles.spinner} size={24} /> Loading personas...</div>;

    return (
        <div className="fade-in">
            {error && <div style={{ ...styles.alert, ...styles.alertDanger, marginTop: '1rem' }}>{error}</div>}
            {(isCreating || editingPersona) &&
                <PersonaForm
                    persona={editingPersona}
                    onSave={handleSave}
                    onCancel={() => { setEditingPersona(null); setIsCreating(false); }}
                />
            }
            <div style={styles.card}>
                <div style={{ ...styles.cardHeader, justifyContent: 'space-between' }}>
                    <div>Your AI Personas</div>
                    <button onClick={() => setIsCreating(true)} style={styles.buttonPrimary}><PlusSquare size={16} /> Create New Persona</button>
                </div>
                {personas.length === 0 ? (
                    <p style={styles.p}>No personas created yet. Click the button above to create one.</p>
                ) : (
                    personas.map(p => (
                        <div key={p.id} style={styles.personaItem}>
                            <div>
                                <h3 style={styles.personaName}>{p.name}</h3>
                                <p style={styles.personaPrompt}><strong>Text:</strong> {p.prompt?.substring(0, 120)}{p.prompt?.length > 120 ? '...' : ''}</p>
                                <p style={styles.personaPrompt}><strong>Voice:</strong> {p.voice_prompt?.substring(0, 120)}{p.voice_prompt?.length > 120 ? '...' : ''}</p>
                                {p.stages && p.stages.length > 0 && <div style={styles.personaStages}>Stages: {p.stages.join(', ')}</div>}
                            </div>
                            <div style={styles.personaActions}>
                                <button onClick={() => setEditingPersona(p)} style={styles.buttonSecondary}><FileEdit size={16} /> Edit</button>
                                <button onClick={() => handleDelete(p.id)} style={styles.buttonDangerOutline}><Trash2 size={16} /> Delete</button>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};


// ==============================================================================
// Compliance Profile Form (Standalone Modal Component)
// ==============================================================================
const ComplianceProfileForm = ({ profile, onSave, onCancel }) => {
    const [name, setName] = useState(profile?.name || '');
    const [rules, setRules] = useState(profile?.content || '');
    const [isSaving, setIsSaving] = useState(false);
    const isEditing = !!profile?.id;

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!name.trim()) return;
        setIsSaving(true);
        onSave({ ...profile, name, content: rules }).finally(() => setIsSaving(false));
    };

    return (
        <div style={styles.modalOverlay}>
            <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
                <div style={styles.modalHeader}>
                    <h2 style={styles.modalTitle}>{isEditing ? 'Edit' : 'Create'} Compliance Profile</h2>
                    <button onClick={onCancel} style={styles.modalCloseButton}><X size={20} /></button>
                </div>
                <form onSubmit={handleSubmit} style={styles.modalBody}>
                    <div style={styles.formGroup}>
                        <label style={styles.label}>Profile Name</label>
                        <input
                            className="modal-input"
                            style={styles.input}
                            value={name}
                            onChange={e => setName(e.target.value)}
                            placeholder="e.g., Financial Services Compliance"
                            required
                        />
                    </div>
                    <div style={styles.formGroup}>
                        <label style={styles.label}>Compliance Rules</label>
                        <textarea
                            className="modal-input"
                            style={styles.textarea}
                            value={rules}
                            onChange={(e) => setRules(e.target.value)}
                            rows={8}
                            placeholder={"Enter rules one per line, with a weight.\nExample:\nDo not give financial advice, 95%\nDo not discuss politics, 80%"}
                        />
                        <p style={styles.formHelperText}>Each line is a rule. End the line with a comma and a percentage weight (e.g., ", 90%") to indicate importance.</p>
                    </div>
                </form>
                <div style={styles.modalFooter}>
                    <button type="button" onClick={onCancel} style={styles.buttonSecondary}>Cancel</button>
                    <button type="button" onClick={handleSubmit} style={styles.buttonSuccess} disabled={isSaving || !name.trim()}>
                        {isSaving ? <><Loader2 style={styles.spinner} size={16} /> Saving...</> : <><Save size={16} /> Save Profile</>}
                    </button>
                </div>
            </div>
        </div>
    );
};

// ==============================================================================
// Compliance Profile Manager
// ==============================================================================
const ComplianceManager = ({ currentUser }) => {
    const [profiles, setProfiles] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');
    const [editingProfile, setEditingProfile] = useState(null);
    const [isCreating, setIsCreating] = useState(false);

    const fetchProfiles = useCallback(async () => {
        setIsLoading(true);
        try {
            const res = await axios.get(`${RAG_BACKEND_URL}/api/compliance`);
            setProfiles(res.data);
        } catch (err) {
            setError('Failed to fetch compliance profiles.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchProfiles();
    }, [fetchProfiles]);

    const handleSave = async (profileData) => {
        try {
            const payload = {
                id: profileData.id || generateUUID(),
                name: profileData.name,
                content: profileData.content
            };
            await axios.post(`${RAG_BACKEND_URL}/api/compliance`, payload);
            setEditingProfile(null);
            setIsCreating(false);
            fetchProfiles();
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to save profile.');
        }
    };

    const handleDelete = async (profileId) => {
        if (window.confirm('Are you sure you want to delete this compliance profile? This will also unassign it from any Knowledge Base.')) {
            try {
                await axios.delete(`${RAG_BACKEND_URL}/api/compliance/${profileId}`);
                fetchProfiles();
            } catch (err) {
                setError('Failed to delete profile.');
            }
        }
    };

    if (isLoading) return <div style={styles.loadingContainer}><Loader2 style={styles.spinner} size={24} /> Loading profiles...</div>;

    return (
        <div className="fade-in">
            {error && <div style={{ ...styles.alert, ...styles.alertDanger, marginTop: '1rem' }}>{error}</div>}

            {(isCreating || editingProfile) &&
                <ComplianceProfileForm
                    profile={editingProfile}
                    onSave={handleSave}
                    onCancel={() => { setEditingProfile(null); setIsCreating(false); }}
                />
            }

            <div style={styles.card}>
                <div style={{ ...styles.cardHeader, justifyContent: 'space-between' }}>
                    <div>Your Compliance Profiles</div>
                    <button onClick={() => setIsCreating(true)} style={styles.buttonPrimary}><PlusSquare size={16} /> Create New Profile</button>
                </div>
                {profiles.length === 0 ? (
                    <p style={styles.p}>No compliance profiles created yet. Click the button to create one.</p>
                ) : (
                    profiles.map(p => (
                        <div key={p.id} style={styles.personaItem}>
                            <div>
                                <h3 style={styles.personaName}>{p.name}</h3>
                                <pre style={styles.complianceContentPreview}>{p.content?.substring(0, 200)}{p.content?.length > 200 ? '...' : ''}</pre>
                            </div>
                            <div style={styles.personaActions}>
                                <button onClick={() => setEditingProfile(p)} style={styles.buttonSecondary}><FileEdit size={16} /> Edit</button>
                                <button onClick={() => handleDelete(p.id)} style={styles.buttonDangerOutline}><Trash2 size={16} /> Delete</button>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

export default DashboardPage;
