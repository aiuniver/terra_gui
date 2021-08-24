<template>
  <form class="block-footer" @submit.prevent>
    <div class="block-footer__item">
      <t-input v-model="nameProject" parse="[name]" small>
        Название датасета
      </t-input>
    </div>
    <div class="block-footer__item block-tags">
      <TTags />
    </div>
    <div class="block-footer__item">
      <Slider :degree="degree"/>
    </div>
    <div class="block-footer__item">
      <t-checkbox parse="[info][shuffle]" reverse>Сохранить последовательность</t-checkbox>
    </div>
    <div class="block-footer__item">
      <t-checkbox parse="use_generator">Использовать генератор</t-checkbox>
    </div>
    <div class="action">
      <t-button :disabled="!!disabled" @click.native="getObj">Сформировать</t-button>
    </div>
  </form>
</template>

<script>
// import DoubleSlider from '@/components/forms/DoubleSlider';
import Slider from "@/components/forms/Slider";
import TTags from '@/components/forms/TTags';
import serialize from "@/assets/js/serialize";
export default {
  name: 'BlockFooter',
  components: {
    Slider,
    TTags,
  },
  data: () => ({
    degree: 100,
    nameProject: 'Новый'
  }),
  computed: {
    disabled() {
      const arr = this.$store.state.datasets.inputData.map(item => item.layer)
      return !(this.nameProject && arr.includes('input') && arr.includes('output'))
    }
  },
  methods: {
    getObj() {
      this.$emit('create', serialize(this.$el))
    }
  }
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
  }
}


.action {
  font-size: 0.8rem;
  button {
    padding: 8px 16px;
  }
}

</style>