<template>
  <form class="block-footer" @submit.prevent>
    <div class="block-footer__item">
      <t-input parse="[name]" small :value="'Новый'">
        Название датасета
      </t-input>
    </div>
    <div class="block-footer__item block-tags">
      <TTags />
    </div>
    <div class="block-footer__item">
      <DoubleSlider />
    </div>
    <div class="block-footer__item">
      <t-checkbox parse="subsequence">Сохранить последовательность</t-checkbox>
    </div>
    <div class="block-footer__item">
      <t-checkbox parse="use_generator">Использовать генератор</t-checkbox>
    </div>
    <div class="action">
      <t-button @click.native="getObj">Сформировать</t-button>
    </div>
  </form>
</template>

<script>
import DoubleSlider from '@/components/forms/DoubleSlider';
import TTags from '@/components/forms/TTags';
import serialize from "@/assets/js/serialize";
export default {
  name: 'BlockFooter',
  components: {
    DoubleSlider,
    TTags,
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