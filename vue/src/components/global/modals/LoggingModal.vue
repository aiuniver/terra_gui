<template>
  <at-modal v-model="dialog" width="800" class="logging-modal">
    <div slot="header">
      <span class="logging-modal__title">{{ title }}</span>
    </div>
    <div class="t-tags">
      <template v-for="tag of tags">
        <div
          :key="tag"
          :class="['t-tags__tag', { 't-tags__tag--active': !selected.includes(tag) }]"
          @click="onChange(tag)"
        >
          {{ tag }}
        </div>
      </template>
    </div>
    <div class="t-logging">
      <div class="t-logging__item">
        <div class="t-logging__date">Время</div>
        <div class="t-logging__type">Тип</div>
        <div class="t-logging__error">Название</div>
      </div>
      <scrollbar>
        <div>
          <template v-for="(error, i) of filter">
            <div class="t-logging__item" :key="'errors_' + i">
              <div class="t-logging__date">{{ (error.time * 1000) | formatDate }}</div>
              <div class="t-logging__type">{{ error.level }}</div>
              <div :class="['t-logging__error', { 't-logging__error': isActive(error) }]" @click="click(error)">
                {{ error.title }}
              </div>
            </div>
          </template>
          <p v-if="!filter.length" class="t-logging__empty">Ничего не выбрано</p>
        </div>
      </scrollbar>
    </div>
    <div slot="footer"></div>
  </at-modal>
</template>

<script>
import { mapActions } from 'vuex';
import filter from '@/mixins/datasets/filters';
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
    selected: [],
  }),
  computed: {
    tags() {
      return [...new Set(this.errors.map(i => i.level))];
    },
    filter() {
      return this.errors.filter(item => !this.selected.includes(item.level));
    },
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
  methods: {
    ...mapActions({
      getLogs: 'logging/get',
    }),
    click(error) {
      // console.log(error.massage);
      if (error.massage) {
        this.$emit('error', error);
      }
    },
    isActive({ message }) {
      return Boolean(message);
    },
    onChange(tag) {
      if (this.selected.includes(tag)) {
        this.selected = this.selected.filter(i => i !== tag);
      } else {
        this.selected.push(tag);
      }
    },
  },
  watch: {
    dialog() {
      this.getLogs();
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
    align-items: center;
    border-bottom: 1px solid #4d5c6a;
    width: 100%;
  }
  &__date {
    width: 100px;
    height: 100%;
  }
  &__type {
    width: 70px;
  }
  &__empty {
    text-align: center;
    margin-top: 50px;
  }
  &__error {
    flex: 1 1 auto;
    white-space: pre-wrap;

    // width: 586px;
    // overflow: hidden;
    &--active {
      cursor: pointer;
      &:hover {
        opacity: 0.7;
      }
    }
  }
}
.t-tags {
  display: flex;
  margin-bottom: 10px;
  &__tag {
    color: #a7bed3;
    border: 1px solid #6c7883;
    background-color: #242f3d;
    border-radius: 4px;
    padding: 0 15px;
    margin-right: 5px;
    cursor: pointer;
    &--active {
      border-color: #65b9f4;
    }
  }
}
</style>
