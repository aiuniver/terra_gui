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
      let text = this.$el.querySelector('.message').innerHTML;
      var textArea = document.createElement('textarea');
      textArea.value = text;

      textArea.style.top = '0';
      textArea.style.left = '0';
      textArea.style.position = 'fixed';

      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();

      try {
        document.execCommand('copy');
        this.copy = true
      } catch (err) {
        console.error('Fallback: Oops, unable to copy', err);
      }
      document.body.removeChild(textArea);
      // alert("Скопировано в буфер обмена");
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
  padding-bottom: 10px;
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
    color: #3eba31;
  }
}
</style>