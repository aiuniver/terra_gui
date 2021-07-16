<template>
  <div class="texts">
    <div class="texts__title">Текст</div>
    <div class="texts__content">
      <div class="inner">
        <div class="epochs">
          <Table :epochs="tables.epochs" :summary="tables.summary" />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
import Table from './Table.vue'
export default {
  name: 'Texts',
  components: {
    Table
  },
  computed: {
    ...mapGetters({
      tables: 'trainings/getTexts'
    })
  }
}
</script>

<style lang="scss" scoped>
.texts {
  margin-bottom: 20px;
  &__title {
    user-select: none;
  }
 &__content {
       margin-top: 10px;
       box-sizing: border-box;
 }
}

</style>

