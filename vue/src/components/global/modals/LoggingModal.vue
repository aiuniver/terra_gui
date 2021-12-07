<template>
  <at-modal v-model="dialog" width="800" class="logging-modal">
    <div slot="header">
      <span class="logging-modal__title">{{ title }}</span>
    </div>
    <div class="t-logging">
      <div class="t-logging__item">
        <div class="t-logging__type">Тип</div>
        <div class="t-logging__date">Время</div>
        <div class="t-logging__error">Название</div>
      </div>
      <scrollbar>
        <template v-for="(error, i) of errors">
          <div class="t-logging__item" :key="'errors_' + i">
            <div class="t-logging__type">{{ error.level }}</div>
            <div class="t-logging__date">{{ (error.time * 1000) | formatDate }}</div>
            <div class="t-logging__error" @click="click(error)">{{ error.title }}</div>
          </div>
        </template>
      </scrollbar>
    </div>
    <div slot="footer"></div>
  </at-modal>
</template>

<script>
import filter from '@/mixins/datasets/filters'
export default {
  name: 'CopyModal',
  mixins: [filter],
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
    click(error) {
      this.$emit('error', error);
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
.logging-modal {
  &__title {
    
  }
}
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
    width: 100px;
  }
  &__type {
    width: 50px;
  }
  &__error {
    cursor: pointer;
    width: 586px;
    overflow: hidden;
    &:hover {
      opacity: 0.7;
    }
  }
}
</style>
