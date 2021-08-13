<template>
  <div :class="['t-field', { 't-inline': inline }]">
    <label class="t-field__label" :for="parse">{{ label }}</label>
    <div class="tags">
      <button class="tags__add" type="button">
        <i class="tags__add--icon t-icon icon-tag-plus"></i>
        <input type="text" class="tags__add--input" :placeholder="'Добавить'" @keypress.enter="create" />
      </button>
      <input v-for="({ value }, i) in tags" :key="'tag_' + i" :value="value" name="[tags][]" type="text" class="tags__item" />
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
    create(e) {
      const value = e.target.value;
      e.target.value = '';
      if (this.tags.length < 3) {
        this.tags.push({ value });
        this.tags = [...this.tags];
      }
    },
    inputLength(e) {
      e.target.style.width = (e.target.value.length + 1) * 8 + 'px';
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
    line-height: 1.5;
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
  }
}
</style>
