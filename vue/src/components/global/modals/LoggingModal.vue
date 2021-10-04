<template>
  <at-modal v-model="dialog" width="800">
    <div slot="header" style="text-align: center">
      <span>{{ title }}</span>
    </div>
    <div class="t-logging">
      <div class="t-logging__item">
        <div class="t-logging__date">Время</div>
        <div class="t-logging__error">Ошибка</div>
      </div>
      <scrollbar>
        <template v-for="({ date, error }, i) of errors">
          <div class="t-logging__item" :key="'errors_' + i">
            <div class="t-logging__date">{{ date | formatDate }}</div>
            <div class="t-logging__error" @click="$emit('error', { error, date })">{{ error }}</div>
          </div>
        </template>
      </scrollbar>
    </div>
    <div slot="footer">
      <!-- <div class="copy-buffer">
        <i :class="['t-icon', 'icon-clipboard']" :title="'copy'" @click="Copy"></i>
        <p v-if="copy" class="success">Код скопирован в буфер обмена</p>
        <p v-else>Скопировать в буфер обмена</p>
      </div> -->
    </div>
  </at-modal>
</template>

<script>
export default {
  name: 'CopyModal',
  props: {
    errors: {
      type: Array,
      default: () => [],
    },
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
  filters: {
    formatDate: value => {
      const date = new Date(value);
      return date.toLocaleString(['ru-RU'], {
        month: 'short',
        day: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    },
  },
};
</script>

<style scoped lang="scss">
.t-logging {
  height: 300px;
  background: #0e1621;
  border-radius: 4px;
  box-shadow: inset 0 0 3px black;
  padding: 10px;
  &__item {
    display: flex;
    border-bottom: 1px solid #4d5c6a;
  }
  &__date {
    width: 130px;
  }
  &__error {
    cursor: pointer;
    &:hover {
      opacity: 0.7;
    }
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
