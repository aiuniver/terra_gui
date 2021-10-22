<template>
  <div class="project project-create" @click="$emit('addCard')">
    <div class="project-edging">
      <svg width="300" height="196" viewBox="0 0 300 196" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M168.663 16.369L168.806 16.5H169H260.304L280.5 35.2183V166.784L262.801 183.5H121.651L96.2774 166.584L96.1514 166.5H96H26.149L0.5 149.73V16.216L17.1983 0.5H151.306L168.663 16.369Z" stroke="#65B9F4"/>
      </svg>
    </div>
    <div class="project-detail">
        <div class="project-create-plus">
          <svg width="54" height="54" viewBox="0 0 54 54" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M52 27C52 40.8071 40.8071 52 27 52C13.1929 52 2 40.8071 2 27C2 13.1929 13.1929 2 27 2C40.8071 2 52 13.1929 52 27Z" stroke="#65B9F4" stroke-width="3"/>
            <path d="M26.375 15.125H27V38.25H26.375V15.125Z" stroke="#65B9F4" stroke-width="3"/>
            <path d="M15.125 27V26.375H38.25V27H15.125Z" stroke="#65B9F4" stroke-width="3"/>
          </svg>
        </div>
        <div class="project-create-headline">
          <h3>Создать новый проект</h3>
        </div>
      </div>
  </div>
</template>

<script>
export default {
  name: 'CardCreateProject',
}
</script>

<style lang="scss" scoped>
.project{
  cursor: pointer;
  position: relative;
  height: 196px;
  &:hover{
    svg{
      fill: rgba(101, 185, 244, 0.02);
    }
  }

  &-detail{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%,-50%);
    width:165px;
  }
  &-create{
    &-plus{
      height: 54px;
      width: 54px;
    }

    &-headline{
      margin-top: 20px;
      h3{
      font-family: "Open Sans";
      font-style: normal;
      font-weight: normal;
      font-size: 14px;
      color: #A7BED3;
      }
    }
  }
}
</style>