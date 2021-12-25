<template>
  <at-modal v-model="dialog" class="error-modal" :width="isMore ? 700 : 600">
    <div class="error-modal__header">
      <span class="error-modal__level">{{ level }}</span>
      <span class="error-modal__title" :title="title">{{ title }}</span>
    </div>
    <div class="t-pre" :style="style">
      <scrollbar>
        <div class="error-modal__content">
          <pre v-if="isMore" ref="message-modal-copy" class="message" v-html="message"></pre>
          <span v-else class="error-modal__btn" @click="isMore = true">Подробней</span>
        </div>
      </scrollbar>
    </div>
    <div slot="footer" class="error-modal__footer">
      <div class="copy-buffer">
        <i :class="['t-icon', 'icon-clipboard']" :title="'copy'" @click="copy"></i>
        <p v-if="isCopy" :class="{ success: isCopy }">
          {{ isCopy ? 'Код скопирован в буфер обмена' : 'Скопировать в буфер обмена' }}
        </p>
      </div>
      <div>
        <span class="error-modal__time">{{ time | formatDate }}</span>
      </div>
    </div>
  </at-modal>
</template>

<script>
import filter from '@/mixins/datasets/filters';
export default {
  name: 'CopyModal',
  mixins: [filter],
  props: {
    title: {
      type: String,
      default: 'Title',
    },
    level: {
      type: String,
      default: '',
    },
    message: {
      type: String,
      default: '',
    },
    time: {
      type: Number,
      default: 0,
    },
    value: {
      type: Boolean,
      default: false,
    },
  },
  data: () => ({
    isCopy: false,
    isMore: false,
  }),
  methods: {
    copy() {
      try {
        let element = this.$refs['message-modal-copy'];
        let range = document.createRange();
        let selection = window.getSelection();
        range.selectNodeContents(element);
        selection.removeAllRanges();
        selection.addRange(range);
        this.isCopy = true;
        document.execCommand('copy');
      } catch (e) {
        console.error('Fallback: Oops, unable to copy', e);
      }
    },
  },
  computed: {
    dialog: {
      set(value) {
        this.$emit('input', value);
        if (!value) {
          this.isCopy = value;
        }
      },
      get() {
        return this.value;
      },
    },
    style() {
      return { height: this.isMore ? '500px' : '60px' };
    },
  },
  watch: {
    dialog(value) {
      if (!value) this.isMore = false;
    },
  },
};
</script>

<style scoped lang="scss">
.error-modal {
  &__header {
    display: flex;
    align-items: flex-end;
    padding-bottom: 10px;
    overflow: hidden;
    width: 90%;
  }
  &__time {
    font-size: 12px;
    opacity: 0.8;
  }
  &__level {
    font-size: 16px;
    margin-right: 10px;
    color: snow;
  }
  &__title {
    font-size: 14px;
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  &__footer {
    display: flex;
    width: 100%;
    justify-content: space-between;
    padding: 0 10px;
  }
  &__content {
    // display: flex;
    // justify-content: center;
    // align-items: center;
  }
  &__btn {
    display: block;
    text-align: center;
    margin-top: 10px;
    font-size: 14px;
    cursor: pointer;
    &:hover {
      color: #6395c2;
    }
  }
}
.t-pre {
  height: 500px;
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
