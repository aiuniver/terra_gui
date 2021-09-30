<template>
  <at-modal v-model="dialog" width="600">
    <div slot="header" style="text-align: center">
      <span>{{ title }}</span>
    </div>
    <div class="t-pre">
      <scrollbar>
        <pre ref="message-modal-copy" class="message"><slot></slot></pre>
      </scrollbar>
    </div>
    <div slot="footer">
      <div class="copy-buffer">
        <i :class="['t-icon', 'icon-clipboard']" :title="'copy'" @click="Copy"></i>
        <p v-if="copy" class="success">Код скопирован в буфер обмена</p>
        <p v-else>Скопировать в буфер обмена</p>
      </div>
    </div>
  </at-modal>
</template>

<script>
export default {
  name: 'CopyModal',
  props: {
    title: {
      type: String,
      default: 'Title',
    },
    value: Boolean,
  },
  data: () => ({
    copy: false,
  }),
  methods: {
    Copy() {
      let element = this.$refs['message-modal-copy'],
        range,
        selection;

      try {
        selection = window.getSelection();
        range = document.createRange();
        range.selectNodeContents(element);
        selection.removeAllRanges();
        selection.addRange(range);
        console.log(selection);

        this.copy = true;

        document.execCommand('copy');
      } catch (e) {
        console.error('Fallback: Oops, unable to copy', e);
      }

      // const selection = window.getSelection();
      // const range = document.createRange();
      // console.log(range);

      // range.selectNode(message);
      // selection.removeAllRanges();
      // selection.addRange(range);
      // message.contentEditable = 'true';

      // try {
      //   document.execCommand('copy');
      //   this.copy = true;
      // } catch (err) {
      //   console.error('Fallback: Oops, unable to copy', err);
      // }

      // message.contentEditable = 'false';
    },
  },
  computed: {
    dialog: {
      set(value) {
        this.$emit('input', value);
        if (!value) {
          this.copy = value;
        }
      },
      get() {
        return this.value;
      },
    },
  },
};
</script>

<style scoped lang="scss">
.t-pre {
  height: 300px;
  background: #0e1621;
  border-radius: 4px;
  box-shadow: inset 0 0 3px black;
  pre {
    user-select: text !important;
    // white-space: break-spaces;
    font-family: monospace;
    padding: 10px;
  }
}

.copy-buffer {
  display: flex;
  align-items: center;
  p {
    margin-left: 15px;
    font-size: 0.85rem;
    &.success {
      color: #3eba31;
    }
  }
  .icon-clipboard {
    cursor: pointer;
  }
}
</style>
