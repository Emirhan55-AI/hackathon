-- Aura AI Platform Database Initialization Script
-- This script sets up the database schema for the platform

-- Create database if not exists (handled by Docker environment)
-- CREATE DATABASE IF NOT EXISTS aura_platform;

-- Use the database
\c aura_platform;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    profile_picture_url TEXT,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) DEFAULT 'New Conversation',
    conversation_type VARCHAR(50) DEFAULT 'general', -- general, outfit_planning, style_advice, etc.
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    tokens_used INTEGER DEFAULT 0,
    processing_time_ms INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Fashion items table
CREATE TABLE IF NOT EXISTS fashion_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL, -- shirt, pants, dress, shoes, etc.
    subcategory VARCHAR(100),
    brand VARCHAR(100),
    color VARCHAR(50),
    size VARCHAR(20),
    image_url TEXT,
    image_analysis JSONB DEFAULT '{}', -- Results from visual analysis
    purchase_date DATE,
    price DECIMAL(10, 2),
    description TEXT,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Outfit combinations table
CREATE TABLE IF NOT EXISTS outfits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    occasion VARCHAR(100), -- casual, formal, business, party, etc.
    season VARCHAR(20), -- spring, summer, fall, winter
    weather_conditions VARCHAR(100),
    style_score DECIMAL(3, 2), -- 0.00 to 10.00
    item_ids UUID[] NOT NULL, -- Array of fashion_items.id
    recommendation_data JSONB DEFAULT '{}', -- ML model output
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
    times_worn INTEGER DEFAULT 0,
    last_worn_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_favorite BOOLEAN DEFAULT FALSE
);

-- User preferences table
CREATE TABLE IF NOT EXISTS user_style_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    style_categories TEXT[] DEFAULT '{}', -- minimalist, bohemian, classic, trendy, etc.
    preferred_colors TEXT[] DEFAULT '{}',
    avoided_colors TEXT[] DEFAULT '{}',
    preferred_brands TEXT[] DEFAULT '{}',
    size_preferences JSONB DEFAULT '{}',
    budget_range JSONB DEFAULT '{}', -- {min: 0, max: 1000}
    occasion_preferences JSONB DEFAULT '{}',
    body_type VARCHAR(50),
    sustainability_preference BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Analytics and tracking table
CREATE TABLE IF NOT EXISTS user_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    interaction_type VARCHAR(50) NOT NULL, -- view, like, dislike, share, purchase, etc.
    target_type VARCHAR(50) NOT NULL, -- outfit, item, recommendation, etc.
    target_id UUID NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Model training feedback table
CREATE TABLE IF NOT EXISTS model_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    model_type VARCHAR(50) NOT NULL, -- visual_analysis, outfit_recommendation, conversational_ai
    input_data JSONB NOT NULL,
    model_output JSONB NOT NULL,
    user_feedback JSONB NOT NULL, -- rating, corrections, comments
    feedback_type VARCHAR(50) NOT NULL, -- rating, correction, report
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_fashion_items_user_id ON fashion_items(user_id);
CREATE INDEX IF NOT EXISTS idx_fashion_items_category ON fashion_items(category);
CREATE INDEX IF NOT EXISTS idx_outfits_user_id ON outfits(user_id);
CREATE INDEX IF NOT EXISTS idx_outfits_occasion ON outfits(occasion);
CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id ON user_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_interactions_type ON user_interactions(interaction_type);
CREATE INDEX IF NOT EXISTS idx_model_feedback_user_id ON model_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_model_feedback_type ON model_feedback(model_type);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic updated_at updates
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_fashion_items_updated_at BEFORE UPDATE ON fashion_items 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_outfits_updated_at BEFORE UPDATE ON outfits 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_style_preferences_updated_at BEFORE UPDATE ON user_style_preferences 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for testing (optional)
INSERT INTO users (username, email, password_hash, first_name, last_name) VALUES
('demo_user', 'demo@aura-ai.com', crypt('demo_password', gen_salt('bf')), 'Demo', 'User')
ON CONFLICT (email) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aura_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aura_user;

-- Create views for analytics
CREATE OR REPLACE VIEW user_activity_summary AS
SELECT 
    u.id,
    u.username,
    u.email,
    COUNT(DISTINCT c.id) as total_conversations,
    COUNT(DISTINCT m.id) as total_messages,
    COUNT(DISTINCT fi.id) as total_items,
    COUNT(DISTINCT o.id) as total_outfits,
    AVG(o.user_rating) as avg_outfit_rating,
    MAX(ui.created_at) as last_activity,
    u.created_at as user_since
FROM users u
LEFT JOIN conversations c ON u.id = c.user_id
LEFT JOIN messages m ON c.id = m.conversation_id
LEFT JOIN fashion_items fi ON u.id = fi.user_id
LEFT JOIN outfits o ON u.id = o.user_id
LEFT JOIN user_interactions ui ON u.id = ui.user_id
GROUP BY u.id, u.username, u.email, u.created_at;

COMMENT ON DATABASE aura_platform IS 'Aura AI Platform - Fashion AI Assistant Database';
COMMENT ON TABLE users IS 'User accounts and profiles';
COMMENT ON TABLE conversations IS 'Chat conversations with AI assistant';
COMMENT ON TABLE messages IS 'Individual messages in conversations';
COMMENT ON TABLE fashion_items IS 'User wardrobe items with AI analysis';
COMMENT ON TABLE outfits IS 'Outfit combinations and recommendations';
COMMENT ON TABLE user_style_preferences IS 'User style and preference settings';
COMMENT ON TABLE user_interactions IS 'User behavior tracking for ML improvement';
COMMENT ON TABLE model_feedback IS 'Feedback for AI model training and improvement';
