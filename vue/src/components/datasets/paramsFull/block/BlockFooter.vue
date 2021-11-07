<template>
  <form class="block-footer" @submit.prevent>
    <div class="block-footer__item">
      <t-field :label="'Название датасета'">
        <t-input-new
          class="block-footer__input-custom"
          v-model="nameProject"
          :style="{ width: '150px'}"
          parse="[name]"
          small
          :error="nameError"
          @focus="nameError = ''"
        ></t-input-new>
      </t-field>
    </div>
    <div class="block-footer__item block-tags">
      <TTags />
    </div>
    <div class="block-footer__item">
      <Slider :degree="degree" />
    </div>
    <div class="block-footer__item block-footer__item--checkbox">
      <t-checkbox parse="[info][shuffle]" reverse inline>Сохранить последовательность</t-checkbox>
      <t-checkbox parse="use_generator" inline>Использовать генератор</t-checkbox>
    </div>
    <div class="action">
      <t-button :disabled="!!disabled" @click.native="getObj">Сформировать</t-button>
    </div>
  </form>
</template>

<script>
// import DoubleSlider from '@/components/forms/DoubleSlider';
import Slider from '@/components/forms/Slider';
import TTags from '@/components/forms/TTags';
import serialize from '@/assets/js/serialize';
export default {
  name: 'BlockFooter',
  components: {
    Slider,
    TTags,
  },
  data: () => ({
    degree: 100,
    nameProject: '',
    nameError: '',
  }),
  computed: {
    disabled() {
      const arr = this.$store.state.datasets.inputData.map(item => item.layer);
      return !(arr.includes('input') && arr.includes('output'));
    },
  },
  methods: {
    getObj() {
      if (this.nameProject) {
        this.$emit('create', serialize(this.$el));
      } else {
        this.nameError = 'Введите имя';
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.block-footer {
  display: flex;
  justify-content: space-between;
  padding: 22px 24px;
  &__item {
    flex: 0 0 150px;
    margin-right: 36px;
    &--checkbox{
      flex: 0 0 250px;
      margin: 5px 0 0 0
    }
  }
  input {
    width: 100%;
  }
}

.action {
  font-size: 0.8rem;
  button {
    padding: 8px 16px;
  }
}
</style>
