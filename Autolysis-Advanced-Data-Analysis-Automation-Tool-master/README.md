# Autolysis: Advanced Data Analysis Automation Tool

## 📊 Project Overview

Autolysis is a sophisticated Python-based data analysis automation tool designed to transform raw CSV datasets into comprehensive, insights-driven reports. By leveraging advanced statistical techniques, machine learning, and large language model (LLM) technologies, Autolysis provides deep, actionable insights from your data with minimal manual intervention.

## 🌟 Key Features

### 1. Comprehensive Data Exploration
- Automatic data loading and preprocessing
- Detailed statistical summaries
- Missing value detection and analysis
- Data type identification

### 2. Advanced Visualization
- Multiple visualization techniques:
  - Histograms
  - Boxplots
  - Correlation heatmaps
  - Feature importance charts
  - Time series analysis (when applicable)

### 3. Intelligent Analysis
- Statistical correlation analysis
- Machine learning-based feature importance
- Random Forest regression for insights
- Adaptive analysis based on dataset characteristics

### 4. AI-Generated Narrative
- LLM-powered README generation
- Contextual insights and interpretations
- Structured markdown reports
- Potential implications and recommendations

## 🛠 Technical Architecture

### Components
- Data Loading: Pandas
- Statistical Analysis: Scikit-learn
- Visualizations: Matplotlib, Seaborn
- LLM Integration: OpenAI API
- Environment Management: python-dotenv

### Workflow
1. Load CSV dataset
2. Perform statistical analysis
3. Generate visualizations
4. Create AI-powered narrative
5. Output comprehensive report

## 🚀 Installation

### Prerequisites
- Python 3.11+
- `uv` or `pip` package manager

### Dependencies
```bash
uv pip install httpx pandas seaborn matplotlib openai scikit-learn python-dotenv pytest-shutil
```

### Environment Setup
1. Clone the repository
2. Create a `.env` file
3. Add your API token:
   ```
   AIPROXY_TOKEN=your_token_here
   ```

## 🔍 Usage

### Basic Execution
```bash
uv run autolysis.py <path_to_csv_file>
```

### Example Datasets
- Goodreads book data analysis
- Global happiness index
- Media popularity metrics

### Sample Command
```bash
uv run autolysis.py goodreads/goodreads.csv
```

## 📂 Repository Structure
```
├── autolysis.py          # Main analysis script
├── LICENSE               # MIT License
├── .env                  # API token configuration
│
├── goodreads/
│   ├── goodreads.csv     # Sample dataset
│   └── README.md         # Generated analysis report
│
├── happiness/
│   ├── happiness.csv     # Sample dataset
│   └── README.md         # Generated analysis report
│
└── media/
    ├── media.csv         # Sample dataset
    └── README.md         # Generated analysis report
```

## 🔬 Supported Analysis Types
- Descriptive Statistics
- Correlation Analysis
- Feature Importance
- Time Series Trends
- Missing Data Patterns

## 🤖 AI Integration
- Uses GPT models for narrative generation
- Contextual insight extraction
- Human-readable report formatting

## 🔒 Security Considerations
- API token managed via environment variables
- No sensitive data stored in code
- Secure API request handling

## 🚧 Limitations
- Requires valid API token
- Performance depends on dataset complexity
- CSV format requirements

## 🔮 Future Roadmap
- Support for more file formats
- Enhanced machine learning models
- Advanced visualization techniques
- Expanded LLM integration

## 📜 License
MIT License - See `LICENSE` file for details

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support
For issues or questions, please open a GitHub issue or contact the maintainer.

---

**Disclaimer**: Autolysis is an analytical tool and should be used as a supplement to, not a replacement for, expert data analysis and interpretation.
