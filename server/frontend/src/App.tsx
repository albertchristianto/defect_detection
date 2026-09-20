import { useState } from 'react'

export default function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [result, setResult] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setImageFile(file)
      setSelectedImage(URL.createObjectURL(file))
      setResult(null)
    }
  }

  const handleImageClick = () => {
    document.getElementById('file-input')?.click()
  }

  const handleDetect = async () => {
    if (!imageFile) return
    
    setLoading(true)
    const formData = new FormData()
    formData.append('file', imageFile)

    try {
      const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setResult(data.result)
    } catch (error) {
      console.error('Detection failed:', error)
      alert('Detection failed')
    } finally {
      setLoading(false)
    }
  }

  const getBorderColor = () => {
    if (!result) return 'transparent'
    return result === 'Defect' ? 'red' : 'green'
  }

  return (
    <div className="app">
      <div className="container">
        <div 
          className="image-area"
          onClick={handleImageClick}
          style={{ borderColor: getBorderColor() }}
        >
          {selectedImage ? (
            <img src={selectedImage} alt="Selected" />
          ) : (
            <div className="placeholder">Click to select image</div>
          )}
          <input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={handleImageSelect}
            style={{ display: 'none' }}
          />
        </div>
        <button 
          className="detect-button"
          onClick={handleDetect}
          disabled={!imageFile || loading}
        >
          {loading ? 'Detecting...' : 'Detect'}
        </button>
        {result && (
          <div className="result">
            Result: {result}
          </div>
        )}
      </div>
    </div>
  )
}
