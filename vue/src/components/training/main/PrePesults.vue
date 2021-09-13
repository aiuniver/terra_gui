<template>
  <div class="pre-results">
    <div class="pre-results__params">
      <div class="params__title">Параметры вывода результатов</div>
      <div class="params__form">
        <div class="form__fields">
          <div class="fields__col">
            <t-checkbox
              :value="false"
              :label="'Выводить промежуточные результаты'"
              type="checkbox"
              :name="'pre-result-print'"
              :inline="true"
            />
          </div>
          <div class="fields__col">
            <t-select
              :value="'val'"
              :label="'Данные для расчета'"
              :lists="['val']"
              :name="'calculate-data'"
              :maxLabel="400"
            />
            <t-select
              :value="'Seed'"
              :label="'Тип выбора данных'"
              :lists="['Seed']"
              :name="'data-type'"
              :maxLabel="400"
            />
          </div>
          <div class="fields__col">
            <t-input
              :value="10"
              :label="'Показать примеров'"
              :type="'number'"
              :name="'num-example'"
              :inline="true"
              :small="true"
            />
            <t-checkbox
              :value="true"
              :label="'Показать статистику'"
              type="checkbox"
              :name="'static-show'"
              :inline="true"
            />
          </div>
          <div class="fields__col">
            <t-checkbox
              :value="false"
              :label="'Автообновление'"
              type="checkbox"
              :name="'autoupdate'"
              :inline="true"
            />
          </div>
        </div>
        <div class="form__btn">
          <button>Показать</button>
        </div>
      </div>
    </div>
    <div class="pre-results__content"></div>
  </div>
</template>

<script>
export default {
  name: "PrePesults"
}
</script>

<style scoped lang="scss">
.params{
  &__title{
    font-size: 14px;
    line-height: 17px;
    color: #A7BED3;
  }
  &__form{
    padding-top: 10px;
    display: flex;
    width: 100%;
  }
}
.form{
  &__fields{
    display: flex;
  }
  &__btn{
    width: 150px;
  }
}
.fields{
  &__col{
    display: flex;
    flex-direction: column;
    padding-top: 10px;
    div{
      width: 100%;
    }
  }
}

</style>