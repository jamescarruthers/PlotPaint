// CodeMirror 6 editor setup for the PlotPaint demo.

import { EditorView, basicSetup } from 'codemirror';
import { keymap } from '@codemirror/view';
import { javascript } from '@codemirror/lang-javascript';
import { oneDark } from '@codemirror/theme-one-dark';

export function createEditor({ parent, doc, onChange, onRun }) {
  const changeListener = EditorView.updateListener.of((update) => {
    if (update.docChanged && onChange) onChange(update.state.doc.toString());
  });

  const runKeys = keymap.of([
    {
      key: 'Mod-Enter',
      run: () => {
        onRun?.();
        return true;
      }
    }
  ]);

  const view = new EditorView({
    doc,
    extensions: [basicSetup, javascript(), oneDark, runKeys, changeListener],
    parent
  });

  return {
    view,
    getCode: () => view.state.doc.toString(),
    setCode: (code) => {
      view.dispatch({
        changes: { from: 0, to: view.state.doc.length, insert: code }
      });
    },
    destroy: () => view.destroy()
  };
}
