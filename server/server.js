const express = require('express');
const app = express();
const dotenv = require('dotenv');
dotenv.config();
const port = process.env.PORT || 3001;
const cors = require('cors');
const authRoutes = require('./routes/authRoutes');
const llmRoutes = require('./routes/llmResponseTest');
const feedbackRoutes = require('./routes/feedbackRoutes');


// Middleware
app.use(express.json());
app.use(cors());
app.use(express.urlencoded({ extended: true }));

// Database connection
const connectDB = require('./config/db');
connectDB();


// Sample route
app.get('/', (req, res) => {
  res.send('Backend server is running!');
});

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/llm', llmRoutes);
app.use('/api/feedback', feedbackRoutes);




// Start server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});