import React, { useState } from 'react';
import axios from 'axios';
import './textapp.css';

const TextApp = () => {
  const [text, setText] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [corpus, setCorpus] = useState('shakespeare');

  const fetchSuggestions = async (inputText, s) => {
    try {
        const response = await axios.post('http://localhost:8000/suggest', {
          text: inputText,
          corpus: corpus
        });
        setSuggestions(response.data.suggestions);
      } catch (error) {
        console.error('Error fetching suggestions:', error);
      }
      console.log('suggestions', suggestions);
    };

  const handleTextChange = async (newText) => {
    console.log('newText', newText);
    setText(newText);
    await fetchSuggestions(newText);
    };

  const handleCorpusChange = async (newCorpus) => {
    console.log('newCorpus', newCorpus);
    setCorpus(newCorpus);
    fetchSuggestions(text);
    };

  return (
    <div className='app'>
        <div className='corpus_selector'>
            <select value={corpus} onChange={(e) => handleCorpusChange(e.target.value)}>
                <option value="shakespeare">Shakespeare</option>
                <option value="kjbible">King James Bible</option>
                <option value="poe">Edgar Allan Poe</option>
                <option value="comte">Auguste Comte</option>
            </select>
        </div>
        <div className='text_editor'>
          <textarea
            className="text-input"
            value={text}
            onChange={(e) => handleTextChange(e.target.value)}
            placeholder="Start typing..."
          />
          {suggestions.length > 0 && (
            <div className="suggestions-box">
              <ul className="suggestions-list">
              <p>what about "<strong>{suggestions}</strong>"a next?</p>
              </ul>
            </div>
          )}
        </div>
    </div>
  );
};

export default TextApp;

