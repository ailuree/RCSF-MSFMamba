import { useEffect } from 'react'
import { X, ZoomIn } from 'lucide-react'

function ImageLightbox({ item, onClose }) {
  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  if (!item) {
    return null
  }

  return (
    <div className="lightbox-backdrop" role="presentation" onClick={onClose}>
      <div
        className="lightbox-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={item.title}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="lightbox-header">
          <div>
            <span>高清预览</span>
            <strong>{item.title}</strong>
            {item.note ? <p>{item.note}</p> : null}
          </div>
          <button type="button" className="lightbox-close-button" onClick={onClose} aria-label="Close image preview">
            <X size={18} />
          </button>
        </div>
        <div className="lightbox-image-shell">
          <img src={item.src} alt={item.title} />
        </div>
        <div className="lightbox-footer">
          <span>
            <ZoomIn size={14} />
            点击遮罩或按 `Esc` 关闭
          </span>
        </div>
      </div>
    </div>
  )
}

export default ImageLightbox
