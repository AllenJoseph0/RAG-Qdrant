import React, { useState } from 'react';
import { BrainCircuit } from 'lucide-react';
import LoginForm from './LoginForm';
import RegisterForm from './RegisterForm';
import './Auth.css';

const AuthModal = ({ onLoginSuccess }) => {
    const [isRegistering, setIsRegistering] = useState(false);

    const toggleMode = () => setIsRegistering(!isRegistering);

    return (
        <div className="auth-overlay">
            <div className="auth-modal">
                <div className="auth-header">
                    <BrainCircuit size={48} style={{ color: '#3b82f6', marginBottom: '1rem' }} />
                    <h2 className="auth-title">
                        {isRegistering ? "Create Account" : "Welcome Back"}
                    </h2>
                    <p className="auth-subtitle">
                        {isRegistering ? "Enter your details to get started" : "Please login to continue"}
                    </p>
                </div>

                {!isRegistering ? (
                    <LoginForm onLoginSuccess={onLoginSuccess} onSwitchToRegister={toggleMode} />
                ) : (
                    <RegisterForm onSwitchToLogin={toggleMode} />
                )}
            </div>
        </div>
    );
};

export default AuthModal;
