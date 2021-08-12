<template>
  <div :class="['t-field', { 't-inline': inline }]">
    <label class="t-field__label" :for="parse">{{ label }}</label>
    <div class="tags">
      <button class="tags__add" type="button"  @click="tagsCount++">
        <i class="t-icon icon-tag-plus"></i>
        <span>Добавить</span>
      </button>
      <input type="text" class="tags__item" @keydown="inputLength" v-for="tag in tagsCount" :key="tag" />
      <!-- <input
      v-model="input"
      class="t-field__input"
      :id="parse"
      :type="type"
      :name="parse"
      :value="value"
      @blur="change"
      :disabled="disabled"
    /> -->
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
    tagsCount: 0,
  }),
  methods: {
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
  .tags {
    display: flex;
    font-size: 12px;
    max-width: 400px;
    flex-wrap: wrap;
    &__item {
      margin-left: 10px;
      padding: 2px 4px;
      width: 60px;
      color: #a7bed3;
      height: 24px;
      color: #ffffff;
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
      span {
        padding-left: 9px;
      }
    }
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
</style>
