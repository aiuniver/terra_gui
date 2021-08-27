<template>
  <div :class="['t-field', { 't-inline': inline }]">
    <label class="t-field__label" :for="parse">{{ label }}</label>
    <div class="tags">
      <button class="tags__add" type="button">
        <i class="tags__add--icon t-icon icon-tag-plus" @click="create"></i>
        <input
          type="text"
          class="tags__add--input"
          :disabled="tags.length >= 3"
          :placeholder="'Добавить'"
          @keypress.enter.prevent="create"
        />
      </button>
      <template v-for="({ value }, i) in tags">
        <input
          :key="'tag_' + i"
          :value="value"
          :data-index="i"
          name="[tags][]"
          type="text"
          :class="['tags__item']"
          :style="{ width: (value.length + 1) * 8 + 'px' }"
          @input="change"
          @blur="blur"
        />
        <!-- <span class="tags__add--span" :key="'span_' + i">{{ value }}</span> -->
      </template>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-input',
  props: {
    label: {
      type: String,
      default: 'Теги',
    },
    type: {
      type: String,
      default: 'text',
    },
    value: {
      type: [String, Number],
    },
    parse: String,
    name: String,
    inline: Boolean,
    disabled: Boolean,
  },
  data: () => ({
    tags: [],
  }),
  methods: {
    create() {
      const el = this.$el.getElementsByClassName('tags__add--input')?.[0];
      console.log(el.value);
      if (el.value && el.value.length >= 3 && this.tags.length <= 3) {
        this.tags.push({ value: el.value });
        this.tags = [...this.tags];
        el.value = '';
      }
    },
    change(e) {
      const index = e.target.dataset.index;
      console.log(index);
      this.tags[+index].value = e.target.value;
    },
    blur(e) {
      const index = e.target.dataset.index;
      if (e.target.value.length <= 2) {
        this.tags = this.tags.filter((item, i) => i !== +index);
      }
      console.log(index);
    },
  },
};
</script>

<style lang="scss" scoped>
.t-field {
  // margin-bottom: 20px;
  &__label {
    text-align: left;
    color: #a7bed3;
    display: block;
    padding-bottom: 10px;
    line-height: 1;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
  }
  &__input {
    color: #fff;
    border-color: #6c7883;
    background: #242f3d;
    height: 42px;
    padding: 0 10px;
    font-size: 0.875rem;
    font-weight: 400;
    border-radius: 4px;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
    &:focus {
      border-color: #fff;
    }
  }
}
.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  align-items: center;
  > label {
    width: 150px;
    max-width: 130px;
    padding: 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1.25;
    font-size: 0.75rem;
  }
  > input {
    height: 22px;
    font-size: 12px;
    line-height: 24px;
    width: 100px;
  }
}

.tags {
  display: flex;
  max-width: 400px;
  &__item {
    margin-left: 10px;
    padding: 2px 2px;
    width: 60px;
    height: 24px;
    color: #a7bed3;
    font-size: 12px;
    font-weight: normal;
    line-height: 24px;
    margin-left: 8px;
    text-align: center;
  }
  &__error {
    border-color: red;
  }
  &__add {
    background: #242f3d;
    height: 24px;
    width: 90px;
    padding: 2px 4px;
    box-shadow: none;
    border-color: #6c7883;
    color: #a7bed3;
    display: flex;
    align-items: center;
    &--icon {
      height: 16px;
      width: 16px;
    }
    &--input {
      height: 24px;
      color: #a7bed3;
      border: none;
      background-color: transparent;
      font-size: 12px;
      padding: 0;
      font-weight: normal;
      line-height: 24px;
      margin-left: 8px;
    }
    &--span {
      background: #242f3d;
      height: 24px;
      padding: 2px 8px;
      box-shadow: none;
      color: #a7bed3;
      display: flex;
      align-items: center;
      border: 1px solid #6c7883;
      border-radius: 4px;
      margin-left: 10px;
      font-size: 12px;
      font-weight: normal;
      line-height: 24px;
    }
  }
}
</style>
