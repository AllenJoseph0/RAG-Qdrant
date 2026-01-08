import commonStyles from '../rag.styles';

const dashboardStyles = {
    ...commonStyles,
    dashboardNav: { display: 'flex', gap: '0.5rem', borderBottom: '1px solid var(--border)', marginBottom: '1.5rem', flexWrap: 'wrap' },
    dashboardNavButton: { padding: '0.75rem 1rem', color: 'var(--muted-foreground)', border: 'none', background: 'transparent', cursor: 'pointer', borderBottom: '3px solid transparent', display: 'flex', alignItems: 'center', gap: '0.5rem', transition: 'color 0.2s, border-color 0.2s' },
    dashboardNavButtonActive: { padding: '0.75rem 1rem', border: 'none', background: 'transparent', cursor: 'pointer', borderBottom: '3px solid var(--primary)', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--primary)', fontWeight: 600 },
    categoryHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem 1.25rem' },
    categoryTitle: { fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.5rem' },
    indexStatus: { color: '#fff', padding: '0.2rem 0.6rem', borderRadius: '12px', fontSize: '0.75rem', marginLeft: '1rem', textTransform: 'uppercase', fontWeight: 600 },
    fileList: { listStyle: 'none', padding: '0 1.25rem 1rem', margin: 0 },
    fileListItem: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '1rem',
        padding: '0.75rem 0',
        borderBottom: '1px solid var(--border)',
        color: 'var(--muted-foreground)',
        flexWrap: 'wrap'
    },
    fileInfoMain: {
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        fontWeight: 500,
        color: 'var(--foreground-heavy)',
        flexShrink: 1,
        minWidth: '200px',
        wordBreak: 'break-all'
    },
    fileInfoMeta: {
        display: 'flex',
        alignItems: 'center',
        gap: '1.5rem',
        fontSize: '0.85rem',
        color: 'var(--muted-foreground)',
        flexShrink: 0,
        textAlign: 'right',
        whiteSpace: 'nowrap'
    },
};

export default dashboardStyles;
