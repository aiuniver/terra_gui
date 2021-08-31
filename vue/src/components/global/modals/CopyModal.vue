<template>
  <at-modal v-model="dialog" width="400">
    <div slot="header" style="text-align: center">
      <span>{{ title }}</span>
    </div>
    <div class="t-pre">
      <scrollbar>
        <p class="message"><slot></slot></p>
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
      const message = this.$el.querySelector('.message');
      const selection = window.getSelection();
      const range = document.createRange();

      range.selectNode(message)
      selection.removeAllRanges();
      selection.addRange(range);
      message.contentEditable = 'true'
      
      try {
        document.execCommand('copy');
        this.copy = true;
      } catch (err) {
        console.error('Fallback: Oops, unable to copy', err);
      }

      message.contentEditable = 'false'
    },
  },
  computed: {
    dialog: {
      set(value) {
        this.$emit('input', value);
        if (!value) {
          this.copy = value
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
  height: 400px;
  background: #0e1621;
  padding: 10px;
  border-radius: 4px;
  box-shadow: inset 0 0 3px black;
  p {
    white-space: break-spaces;
    font-family: monospace;
  }
}

.copy-buffer {
  display: flex;
  align-items: center;
  p {
    margin-left: 15px;
    font-size: 0.85rem;
    color: white;
    &.success {
      color: #3eba31;
    }
  }
}
</style>